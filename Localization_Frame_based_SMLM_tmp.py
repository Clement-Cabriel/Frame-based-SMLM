# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:08:57 2020

@author: Clément Cabriel
"""

#-----------------------------------------------------------------------------
# Localization algorithm for Single-Molecule Localization Microscopy
#-----------------------------------------------------------------------------
# *AUTHOR:
#     Clément Cabriel
#     Institut Langevin, ESPCI Paris / CNRS
#     clement.cabriel@espci.fr , cabriel.clement@gmail.com
#     ORCID: 0000-0002-0316-0312
#     Github: https://github.com/Clement-Cabriel
# If you use the code in a publication, please include a reference to the Github repository (https://github.com/Clement-Cabriel/Frame-based-SMLM). If you would like to use the code for commercial purposes, please contact me
#-----------------------------------------------------------------------------
# *VERSION INFORMATION
# Created on November 23rd 2020, last updated on May 16th 2024
# Tested with Python 3.8.3 and Python 3.8.8
#-----------------------------------------------------------------------------
# *CODE DESCRIPTION
# This codes consists of several steps:
#       - The image stack is imported 
#       - The PSFs are detected on each frame through a wavelet filtering
#       - Each PSF is localized by Gaussian fitting
#       - The output data (localization list and SMLM image) are exported
# Note: this code does not perform drift correction. If it is required, consider using a post-processing code taking the localization list as input
# To use the code, set the parametes in the section below, then run the script.
#-----------------------------------------------------------------------------
# *INPUT FORMATS
# .tif image stack with columns in the following order: x,y,t. If the input stack has a different order, consider modifying the 'import_image_stack' function
#-----------------------------------------------------------------------------
# *OUTPUT FORMATS
# "_localization_image.tif": 2D SMLM image (each molecule counts as 1 in the corresponding pixel)
# "_localization_results.npy": file containing the localized coordinates. Each row (axis 0) is one molecule, and the different columns (axis 1) contain the different coordinates:
#       Column 0: frame number
#       Column 1: molecule ID
#       Column 2: y (in optical pixels)
#       Column 3: x (in optical pixels)
#       Column 4: PSF y width (in optical pixels or in nm)
#       Column 5: PSF x width (in optical pixels or in nm)
#       Column 6: amplitude of the fitted Gaussian
#       Column 7: background (i.e. offset of the fitted Gaussian)
#       Column 8: fitting error (defined as err=1-R^2, i.e. a good localization is close to 0, and a bad localization is close to 1)
#       Column 9: photon count (i.e. number of photons intedrated over the whole fitting area, background subtracted)
#    for instance, the element [540,5] of the array is the PSF x witdh of the 540th molecule
#-----------------------------------------------------------------------------

import numpy as np
import skimage.io
import os
import matplotlib.pyplot as plt
import scipy.ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import optimize
import tifffile
import time
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
import copy
import matplotlib as mpl
import gc
import heapq

class Localization_workflow():
    
    def __init__(self):
        
        #-----------------------------------------------------------------------------
        # USER DEFINED PARAMETERS
        #-----------------------------------------------------------------------------
        
        # Input data
        self.folders=['Path/to/image_stack.tif']
        self.frame_limits=[0]                       # Limits of the stack to be processed. =[0] to process everything; =[N_min] to process everything from N_min to the end; =[0,N_max] to process everything from the start to N_max; =[N_min,N_max] to process everything from N_min to N_max
        self.ROI=[0]                                 # Limits of the ROI (in pixels). =[x_min,x_max,y_min,y_max]. Set any value to -1 to keep the default value (for example =[50,-1,-1,-1] to remove the first 50 x pixels). Set =[] to keep all the default parameters
        self.pixel_object_plane=107.                 # Size of the object plane (in nm)
        
        # Filtering parameters
        self.temporal_filter_method="Local median"  # Type of temporal filter. ="None", "Local mean", "Local median"
        self.temporal_filter_half_size=10            # Half size (in frames) of the sliding window used for the local temporal filtering
        self.smoothing_frame_to_label=.5          # Sigma (in pixels) of the Gaussian smoothing applied on the filtered frame before detecting the labels
        
        # Wavelet detection and filtering
        self.threshold_detection=2.75                # Wavelet detection threshold
        self.min_diameter=1.25                        # Minimum radius of the thresholded area (in pixels)
        self.max_diameter=5.0                         # Maximum radius of the thresholded area (in pixels)
        
        # Localization parameters
        self.area_radius=4                          # Radius of the fitting area (in pixels)
        self.allow_ellipticity=True                 # =True to allow x-y asymmetric Gaussian fitting ; =False to restrict the fitting to isotropic Gaussians
        
        # Computation parameters
        self.multithread_processing=True         # =True to use multithread parallelization for the localization (faster, but might require more RAM); =False not to
        
        # Display paramters
        self.print_number_of_molecules=False      # =True to display the number of molecules detected and kept for each frame; =False not to
        self.number_frames_displayed=0              # Number of frames to be displayed. =0 not to display the frames. Note: large number of images (above several tens) may cause crashes
        self.display_centers=False                   # =True to display the localized centers of the PSFs on the frames; =False not to
        self.display_wxwy=False                      # =True to display the localized widths of the PSFs on the frames; =False not to
        
        # Localization file export parameters
        self.distances_pix_or_nm="pix"              # Format of the output distances. ="nm" or "pix" (for camera pixels)
        
        # Image export parameters
        self.generate_localization_image=True      # =True to create the localization images (RCC corrected and uncorrected); =False not to
        self.localization_pixel=15.                 # Size of the localization image pixel (in nm). Used only for the SR maps
        
        #-----------------------------------------------------------------------------
        # END OF USER DEFINED PARAMETERS
        # DEFINITION OF VARIABLES
        #-----------------------------------------------------------------------------
        
        # General
        print('')
        print('Starting')
        print('')
        gc.collect()
        
        # Filtering
        self.temporal_filter_offset=0
        if self.temporal_filter_method=="None":
            self.temporal_filter_half_size=0
            self.temporal_filter_offset=0
        
        # Wavelet
        self.kernel1=np.array([0.0625,0.25,0.375,0.25,0.0625])
        self.kernel2=np.array([0.0625,0,0.375,0,0.25,0,0.0625])
        kernel_size=8
        self.kernel=np.ones((kernel_size,kernel_size))/(kernel_size**2.)
        
        # Display
        self.subplots_titles=["Raw frame","Filtered frame","Detected PSFs"]
        self.cmap_sat_under=copy.copy(mpl.cm.get_cmap("hot"))
        self.cmap_sat_under.set_under(color='gray')
        self.cmaps=["hot","hot",self.cmap_sat_under]
        
        # Localization
        self.fluorescence_wavelength=680.
        self.initial_guess_wx=self.fluorescence_wavelength/4./self.pixel_object_plane
        self.initial_guess_wy=self.fluorescence_wavelength/4./self.pixel_object_plane
        self.yrange=np.arange(0,self.area_radius*2+1)*1.
        self.xrange=np.arange(0,self.area_radius*2+1)*1.
        self.exclusion_radius=int(self.area_radius)
        
        #-----------------------------------------------------------------------------
        # END OF DEFINITION OF VARIABLES
        # MAIN CODE
        #-----------------------------------------------------------------------------
        for f in range(len(self.folders)):
            self.tref=time.time()
            self.folder=self.folders[f]
            print("------------------------")
            print("Processing: "+self.folder)
            print("------------------------")
            self.filename,self.file_extension=os.path.splitext(self.folder)
            self.Main()
            time.sleep(5.)
    
    def import_image_stack(self,path,extension):
        
        stack=tifffile.imread(path+extension)
        self.raw_stack_size=np.shape(stack)
        print('Frame stack shape:','x:',self.raw_stack_size[2],'y:',self.raw_stack_size[1],'t:',self.raw_stack_size[0])
        if len(np.shape(stack))==2:
            stack=np.zeros((1,np.shape(stack)[0],np.shape(stack)[1]),dtype=int)+stack
        self.raw_frame_size=[np.shape(stack)[1],np.shape(stack)[2]]
        filter_ROI,limits=self.define_ROI(np.shape(stack)[1],np.shape(stack)[2])
        if filter_ROI==True:
            stack=stack[:,limits[0]:limits[1],limits[2]:limits[3]]
        
        return stack
    
    def define_ROI(self,y_size,x_size):
        
        if len(self.ROI)==4:
            limits=[]
            filter_ROI=True
            if self.ROI[2]<0:
                limits.append(0)
            else:
                limits.append(self.ROI[2])
            if self.ROI[3]<0:
                limits.append(y_size)
            else:
                limits.append(self.ROI[3])
            if self.ROI[0]<0:
                limits.append(0)
            else:
                limits.append(self.ROI[0])
            if self.ROI[1]<0:
                limits.append(x_size)
            else:
                limits.append(self.ROI[1])
        else:
            limits=[0,y_size,0,x_size]
            filter_ROI=False
        self.filter_ROI=filter_ROI
        self.limits=limits
            
        return filter_ROI,limits
    
    def filter_frame_temporal(self,index):
        
        indices=self.indices_temporal_filter(np.shape(self.stack)[0],self.temporal_filter_offset,self.temporal_filter_half_size,index)
        if self.temporal_filter_method=="None":
            frame_filtered=self.stack[index,:,:]
        elif self.temporal_filter_method=="Global mean":
            frame_filtered=self.stack[index,:,:]-np.round(np.mean(self.stack[:,:,:],axis=0))
        elif self.temporal_filter_method=="Global median":
            frame_filtered=self.stack[index,:,:]-np.round(np.median(self.stack[:,:,:],axis=0))
        elif self.temporal_filter_method=="Local mean":
            frame_filtered=self.stack[index,:,:]-np.round(np.mean(self.stack[indices,:,:],axis=0))
        elif self.temporal_filter_method=="Local median":
            frame_filtered=self.stack[index,:,:]-np.round(np.median(self.stack[indices,:,:],axis=0))
        elif self.temporal_filter_method=="Keep local mean":
            frame_filtered=np.round(np.mean(self.stack[indices,:,:],axis=0))
        elif self.temporal_filter_method=="Keep local median":
            frame_filtered=np.round(np.median(self.stack[indices,:,:],axis=0))
            
        return frame_filtered
    
    def indices_temporal_filter(self,length,offset,half_size,k):
        
        ind_low=max(0,k-(half_size+offset)*2)
        ind_high=min(length,k+(half_size+offset)*2)
        indices=np.arange(ind_low,ind_high).tolist()
        for l in np.arange(offset):
            if k-l in indices:
                indices.remove(k-l)
            if k+l in indices:
                indices.remove(k+l)
            if k-l-offset-half_size in indices:
                indices.remove(k-l-offset-half_size)
            if k+l+offset+half_size in indices:
                indices.remove(k+l+offset+half_size)
            
        return indices
    
    def wavelet_detection(self,frame):
        # Adapted from Izeddin et al. Optics Express, 'Wavelet analysis for single molecule localization microscopy' (2012), DOI: 10.1364/OE.20.002081
        
        V1=scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(frame,self.kernel1,axis=1),self.kernel1,axis=0)
        V2=scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(V1,self.kernel2,axis=1),self.kernel2,axis=0)
        Wavelet2nd=V1-V2
        Wavelet2nd-=scipy.ndimage.convolve(Wavelet2nd,self.kernel)
        Wavelet2nd*=(Wavelet2nd>=0)
        image_to_label=(Wavelet2nd>=self.threshold_detection*np.std(Wavelet2nd))*Wavelet2nd
        return image_to_label
    
    def detect_PSFs(self,frame):
        
        # Generate image to label
        image_to_label=self.wavelet_detection(frame)
        
        # Create labels
        labels,nb_labels=scipy.ndimage.label(image_to_label)
        label_list=np.arange(nb_labels)+1
        area_sizes=scipy.ndimage.sum((image_to_label>0),labels,index=label_list)
        
        # Filter labels by size
        msk=(area_sizes>(self.min_diameter*2+1)**2)*(area_sizes<(self.max_diameter*2+1)**2)
        label_list=label_list[msk]
        labels*=np.isin(labels,label_list)
        nb_labels=len(label_list)
        
        # Calculate center of mass
        coordinates_COM=scipy.ndimage.center_of_mass(image_to_label,labels,label_list)
        coordinates_COM=np.asarray(coordinates_COM)
        if len(np.shape(coordinates_COM))==1:
            if np.shape(coordinates_COM)[0]==0:
                coordinates_COM=np.zeros((0,2))
        
        # Filter coordinates by distance to each other
        msk=np.ones(nb_labels,dtype=bool)
        mgy1,mgy2=np.meshgrid(coordinates_COM[:,0],coordinates_COM[:,0].transpose())
        mgx1,mgx2=np.meshgrid(coordinates_COM[:,1],coordinates_COM[:,1].transpose())
        msk2=(np.abs(mgy1-mgy2)<2*self.exclusion_radius)*(np.abs(mgx1-mgx2)<2*self.exclusion_radius)
        np.fill_diagonal(msk2,False)
        ind=np.nonzero(msk2)
        ind=ind[0].tolist()+ind[1].tolist()
        ind=list(set(ind))
        msk[ind]=False
        label_list=label_list[msk]
        labels*=np.isin(labels,label_list)
        nb_labels=len(label_list)
        coordinates_COM=coordinates_COM[msk]
        
        # Filter coordinates by distance to the frame edges
        if nb_labels>0:
            msk=(coordinates_COM[:,0]>self.area_radius*2)*(coordinates_COM[:,1]>self.area_radius*2)*(coordinates_COM[:,0]<np.shape(frame)[0]-self.area_radius*2)*(coordinates_COM[:,1]<np.shape(frame)[1]-self.area_radius*2)
            label_list=label_list[msk]
            labels*=np.isin(labels,label_list)
            nb_labels=len(label_list)
            coordinates_COM=coordinates_COM[msk]
            
        if len(np.shape(coordinates_COM))==1:
            coordinates_COM=np.zeros((0,2))
        
        return coordinates_COM
    
    def display_frame(self,frame,name,position):
        
        plt.figure(name)
        figManager=plt.get_current_fig_manager()
        figManager.window.showMaximized()
        sbp=plt.subplot(1,3,position+1)
        plt.title(self.subplots_titles[position])
        if not self.cmaps[position]==self.cmap_sat_under:
            ax=sbp.imshow(frame,cmap=self.cmaps[position],interpolation='none',vmin=0.0,vmax=(frame.max()-frame.min())+frame.min())
        else:
            if np.sum((frame>0))>0:
                minval=np.min(frame[frame>0])
            else:
                minval=0
            ax=sbp.imshow(frame,cmap=self.cmaps[position],interpolation='none',vmin=minval+0.001,vmax=minval+0.001*2+(frame.max()-minval)+minval)
        cax=make_axes_locatable(sbp).append_axes("right",size="5%",pad=0.05)
        plt.colorbar(ax,cax=cax,format="%1i")
        
    def display_figure(self,raw_frame,filtered_frame,labelled_image,coordinates,name_figure):
        
        self.display_frame(raw_frame,name_figure,0)
        self.display_frame(filtered_frame,name_figure,1)
        self.display_frame(labelled_image,name_figure,2)
        if self.display_centers==True:
            self.display_centers_figure(coordinates,name_figure,2)
        if self.display_wxwy==True:
            self.display_wxwy_figure(coordinates,name_figure,2)
        
    def display_centers_figure(self,coordinates,name,position):
        
        plt.figure(name)
        plt.subplot(1,3,position+1)
        for k in range(np.shape(coordinates)[0]):
            plt.plot(coordinates[k,1],coordinates[k,0],'xb')
            
    def display_wxwy_figure(self,coordinates,name,position):
        
        plt.figure(name)
        plt.subplot(1,3,position+1)
        for k in range(np.shape(coordinates)[0]):
            plt.plot([coordinates[k,1]-coordinates[k,2]*2.,coordinates[k,1]+coordinates[k,2]*2.],[coordinates[k,0],coordinates[k,0]],'|g',mew=1.5)
            plt.plot([coordinates[k,1],coordinates[k,1]],[coordinates[k,0]-coordinates[k,3]*2.,coordinates[k,0]+coordinates[k,3]*2.],'_g',mew=1.5)
        
    def localization_boxes_image(self,frame,coordinates_COM):
        
        labelled_image=np.zeros((np.shape(frame)[0],np.shape(frame)[1]),dtype=int)
        for k in np.arange(np.shape(coordinates_COM)[0]):
            center=np.round(coordinates_COM[k,:]).astype(int)
            labelled_image[center[0]-self.area_radius:center[0]+self.area_radius+1,center[1]-self.area_radius:center[1]+self.area_radius+1]=1
        frame_copy=frame.copy()
        frame_copy[frame_copy<=0.]=1e-2
        return labelled_image*frame_copy
    
    def localization(self,frame,coordinates_COM):
        
        coordinates=np.zeros((np.shape(coordinates_COM)[0],8))
        for k in np.arange(np.shape(coordinates_COM)[0]):
            center=np.round(coordinates_COM[k,:]).astype(int)
            sub_image=frame[center[0]-self.area_radius:center[0]+self.area_radius+1,center[1]-self.area_radius:center[1]+self.area_radius+1]
            
            if self.allow_ellipticity==False:
                coordinates[k,[0,1,2,4,5]],coordinates[k,6]=self.gaussian_fitting_symmetrical(coordinates_COM[k,0]-(center[0]-self.area_radius),coordinates_COM[k,1]-(center[1]-self.area_radius),sub_image)
                coordinates[k,3]=coordinates[k,2]
            elif self.allow_ellipticity==True:
                coordinates[k,[0,1,2,3,4,5]],coordinates[k,6]=self.gaussian_fitting(coordinates_COM[k,0]-(center[0]-self.area_radius),coordinates_COM[k,1]-(center[1]-self.area_radius),sub_image)
            base_level=np.mean(heapq.nsmallest(int(np.ceil(0.3*np.shape(sub_image)[0]*np.shape(sub_image)[1])),np.ndarray.flatten(sub_image)))
            msk=np.ones((np.shape(sub_image)[0],np.shape(sub_image)[1]),dtype=bool)
            msk[1:-1,1:-1]=False
            base_level=np.mean(sub_image[msk])
            photon_count=np.sum(sub_image-base_level)
            coordinates[k,7]=photon_count
            coordinates[k,0]+=center[0]-self.area_radius
            coordinates[k,1]+=center[1]-self.area_radius
            
        return coordinates
    
    def gaussian_fitting(self,y0,x0,sub_image):
        
        initial_guess=self.area_radius,self.area_radius,self.initial_guess_wx,self.initial_guess_wy,np.max(sub_image)-np.min(sub_image),np.min(sub_image)
        errorfunction=lambda p:np.ravel(self.gaussian(*p)-sub_image)
        p_full = optimize.leastsq(errorfunction,initial_guess,gtol=1e-4,ftol=1e-4,full_output=True)
        p=p_full[0]
        
        # Calculation of the goodness of fitting
        fv=p_full[2]['fvec']
        ss_err=(fv**2).sum()
        ss_tot=((sub_image-sub_image.mean())**2).sum()
        err=ss_err/ss_tot
        
        return p,err
    
    def gaussian(self,y0,x0,widthx,widthy,height,offset):
        
        fX=np.exp(-(self.xrange-x0)**2/(2.*widthx**2))
        fY=np.exp(-(self.yrange-y0)**2/(2.*widthy**2))
        fY=fY.reshape(len(fY),1)
        return offset+height*fY*fX
    
    def gaussian_fitting_symmetrical(self,y0,x0,sub_image):
        
        initial_guess=self.area_radius,self.area_radius,self.initial_guess_wx,np.max(sub_image)-np.min(sub_image),np.min(sub_image)
        errorfunction=lambda p:np.ravel(self.gaussian_symmetrical(*p)-sub_image)
        p_full = optimize.leastsq(errorfunction,initial_guess,gtol=1e-4,ftol=1e-4,full_output=True)
        p=p_full[0]
        fv=p_full[2]['fvec']
        ss_err=(fv**2).sum()
        ss_tot=((sub_image-sub_image.mean())**2).sum()
        err=ss_err/ss_tot
        
        return p,err
    
    def gaussian_symmetrical(self,y0,x0,width,height,offset):
        
        fX=np.exp(-(self.xrange-x0)**2/(2.*width**2))
        fY=np.exp(-(self.yrange-y0)**2/(2.*width**2))
        fY=fY.reshape(len(fY),1)
        return offset+height*fY*fX
    
    def conversion_output(self,coordinates):
        
        if self.filter_ROI==True:
            coordinates[:,0]+=self.limits[0]
            coordinates[:,1]+=self.limits[2]
        
        if self.distances_pix_or_nm=="nm":
            coordinates[:,[0,1,2,3]]*=self.pixel_object_plane
        return coordinates
    
    def generate_images(self,limits,results_localization):
        
        size_image=[int(np.ceil(self.raw_frame_size[0]*self.pixel_object_plane/self.localization_pixel)),int(np.ceil(self.raw_frame_size[1]*self.pixel_object_plane/self.localization_pixel))]
        offset=[self.limits[0],self.limits[2]]
        self.localization_image_2D=np.zeros((size_image[0],size_image[1]))
        
        print("")
        print("Generating localization image")
        for k in np.arange(np.shape(results_localization)[0]):
            if k%10000==0:
                print("Molecule %s/%s (%s "%(k,np.shape(results_localization)[0],100.*k/np.shape(results_localization)[0])+"%)")
            pixel_coordinates=[int(np.floor((offset[0]+results_localization[k,0])*self.pixel_object_plane/self.localization_pixel)),int(np.floor((offset[1]+results_localization[k,1])*self.pixel_object_plane/self.localization_pixel))]
            if pixel_coordinates[0]>=0 and pixel_coordinates[0]<size_image[0]-1 and pixel_coordinates[1]>=0 and pixel_coordinates[1]<size_image[1]-1:
                self.localization_image_2D[pixel_coordinates[0],pixel_coordinates[1]]+=1
                
        return self.localization_image_2D
    
    def Save_parameters_init(self,frame_numbers,frame_sizes):
        
        parameters_dictionary={"folder":self.folder,"frame_limits":self.frame_limits,"ROI":self.ROI,"pixel_object_plane":self.pixel_object_plane,
        "temporal_filter_method":self.temporal_filter_method,"temporal_filter_half_size":self.temporal_filter_half_size,"smoothing_frame_to_label":self.smoothing_frame_to_label,
        "threshold_detection":self.threshold_detection,"min_diameter":self.min_diameter,"max_diameter":self.max_diameter,
        "area_radius":self.area_radius,"allow_ellipticity":self.allow_ellipticity,
        "multithread_processing":self.multithread_processing,
        "number_frames_displayed":self.number_frames_displayed,"print_number_of_molecules":self.print_number_of_molecules,"display_centers":self.display_centers,"display_wxwy":self.display_wxwy,
        "distances_pix_or_nm":self.distances_pix_or_nm,
        "generate_localization_image":self.generate_localization_image,"localization_pixel":self.localization_pixel}
        
        start_time=datetime.now()
        start_time=start_time.strftime("%d/%m/%Y %H:%M:%S")
        run_parameters_dictionary={"Start time":start_time,"frame_numbers":frame_numbers,"initial_stack_shape":self.raw_stack_size,"initial_frame_sizes":self.raw_frame_size,"cropped_frame_sizes":frame_sizes,"ROI limits":self.limits}
        
        with open(self.filename+'_log.txt', 'w') as f:
            print(__file__,file=f)
            print('',file=f)
            print('PARAMETERS',file=f)
            for k in list(parameters_dictionary.keys()):
                print('        self.'+k+'='+str(parameters_dictionary[k]), file=f)
        with open(self.filename+'_log.txt', 'a') as f:
            print('',file=f)
            print('INITIALIZATION LOG',file=f)
            for k in list(run_parameters_dictionary.keys()):
                print(k+'='+str(run_parameters_dictionary[k]), file=f)
    
    def Save_parameters_final(self):
        
        end_time=datetime.now()
        end_time=end_time.strftime("%d/%m/%Y %H:%M:%S")
        final_parameters_dictionary={"End time":end_time}
        
        with open(self.filename+'_log.txt', 'a') as f:
            print('',file=f)
            print('COMPLETION LOG',file=f)
            for k in list(final_parameters_dictionary.keys()):
                print(k+'='+str(final_parameters_dictionary[k]), file=f)
            print('Done',file=f)

    def Main(self):
        
        # Initializations
        print('')
        self.stack=self.import_image_stack(self.filename,self.file_extension)
        print("Stack imported")
        frame_numbers=np.arange(np.shape(self.stack)[0])
        self.Save_parameters_init([frame_numbers[0],frame_numbers[-1]],[np.shape(self.stack)[1],np.shape(self.stack)[2]])
        if self.number_frames_displayed!=0 and len(self.frame_limits)==2:
            pass
        elif len(self.frame_limits)==1:
            frame_numbers=frame_numbers[self.frame_limits[0]:]
        elif len(self.frame_limits)==2:
            frame_numbers=frame_numbers[self.frame_limits[0]:self.frame_limits[1]+1]
        expected_number_of_PFSs_per_frame=np.shape(self.stack)[1]*np.shape(self.stack)[2]/(self.exclusion_radius*2+1.)**2
        expected_number_of_PFSs_per_frame*=.035
        results_localization_block=np.zeros((len(frame_numbers)*int(np.ceil(expected_number_of_PFSs_per_frame)),10))
        results_localization=results_localization_block.copy()
        cnt=0
        cnt_frames=0
        cnt_offset=0
        
        # Localization
        print('')
        print('Starting localization')
            
        if self.multithread_processing==True:
            num_cores = multiprocessing.cpu_count()
        else:
            num_cores = 1
            
        if self.number_frames_displayed>0 or self.multithread_processing==False:
            
            for k in frame_numbers:
                if (k-frame_numbers[0])%100==0 or k==frame_numbers[-1]:
                    print("Frame %s: %s/%s (%s "%(k,k-frame_numbers[0],len(frame_numbers),100.*(k-frame_numbers[0])/len(frame_numbers))+"%)")
                if self.number_frames_displayed==0 or cnt_offset>(self.temporal_filter_offset+self.temporal_filter_half_size+1)*2+1:
                    raw_frame=self.stack[k,:,:]*1.
                    name_figure=self.folder+" - Frame "+str(k)
                    # PSF detection
                    filtered_frame=self.filter_frame_temporal(k).astype(float)
                    filtered_frame_to_label=filtered_frame.copy()
                    if self.smoothing_frame_to_label>0.:
                        filtered_frame_to_label=scipy.ndimage.gaussian_filter(filtered_frame_to_label,self.smoothing_frame_to_label)
                    coordinates_COM=self.detect_PSFs(filtered_frame_to_label)
                    # Localization
                    coordinates_loca=self.localization(raw_frame,coordinates_COM)
                    msk=np.isnan(coordinates_loca)
                    msk=(np.sum(msk,axis=1)==np.shape(msk)[1])
                    coordinates_loca=coordinates_loca[msk==False,:]
                    if self.print_number_of_molecules==True:
                        print('    Frame '+str(k)+' - number of detected molecules: '+str(np.shape(coordinates_loca)[0]))
                    # Results compilation
                    if cnt+np.shape(coordinates_loca)[0]>np.shape(results_localization)[0]-1:
                        results_localization=np.concatenate((results_localization,results_localization_block))
                    results_localization[cnt:cnt+np.shape(coordinates_loca)[0],2:]=coordinates_loca
                    results_localization[cnt:cnt+np.shape(coordinates_loca)[0],1]=cnt+np.arange(np.shape(coordinates_loca)[0])
                    results_localization[cnt:cnt+np.shape(coordinates_loca)[0],0]=k
                    cnt+=np.shape(coordinates_loca)[0]
            
                # Raw frame, filtering and PSF detection display
                if cnt_frames<self.number_frames_displayed and cnt_offset>(self.temporal_filter_offset+self.temporal_filter_half_size+1)*2+1:
                    labelled_image=self.localization_boxes_image(filtered_frame_to_label,coordinates_COM)
                    self.display_figure(raw_frame,filtered_frame_to_label,labelled_image,coordinates_loca,name_figure)
                    cnt_frames+=1
                elif cnt_frames==self.number_frames_displayed:
                    print('')
                    print("Frame display done")
                    cnt_frames+=1
                    if self.number_frames_displayed>0:
                        break
                cnt_offset+=1
                    
            results_localization=results_localization[:cnt,:]
            
        else:
        
            def compute_thread_localization(k):
                
                if (k-frame_numbers[0])%100==0 or k==frame_numbers[-1]:
                    print("Frame %s: %s/%s (%s "%(k,k-frame_numbers[0],len(frame_numbers),100.*(k-frame_numbers[0])/len(frame_numbers))+"%)")
                raw_frame=self.stack[k,:,:]*1.
                # PSF detection
                filtered_frame=self.filter_frame_temporal(k).astype(float)
                filtered_frame_to_label=filtered_frame.copy()
                if self.smoothing_frame_to_label>0.:
                    filtered_frame_to_label=scipy.ndimage.gaussian_filter(filtered_frame_to_label,self.smoothing_frame_to_label)
                coordinates_COM_k=self.detect_PSFs(filtered_frame_to_label)
                # Localization
                coordinates_loca_k=self.localization(raw_frame,coordinates_COM_k)
                msk=np.isnan(coordinates_loca_k)
                msk=(np.sum(msk,axis=1)==np.shape(msk)[1])
                coordinates_loca_k=coordinates_loca_k[msk==False,:]
                coordinates_COM_k=coordinates_COM_k[msk==False,:]
                if self.print_number_of_molecules==True:
                    print(np.shape(coordinates_loca_k)[0])
                coordinates_k=[coordinates_loca_k,coordinates_COM_k]
                
                return coordinates_k
            
            # Results compilation
            tr=time.time()
            RES_loca = Parallel(n_jobs=num_cores,backend="loky")(delayed(compute_thread_localization)(k) for k in frame_numbers)
            print('Localization done in',time.time()-tr)
            nb_loca=0
            for k in range(np.shape(RES_loca)[0]):
                nb_loca+=np.shape(RES_loca[k][1])[0]
            cnt_loca=0
            results_localization=np.zeros((nb_loca,np.shape(results_localization_block)[1]))
            for k in range(np.shape(RES_loca)[0]):
                frame_num=frame_numbers[k]
                cnt_loca_k=np.shape(RES_loca[k][1])[0]
                results_localization[cnt_loca:cnt_loca+cnt_loca_k,2:]=RES_loca[k][0]
                results_localization[cnt_loca:cnt_loca+cnt_loca_k,1]=cnt_loca+np.arange(cnt_loca_k)
                results_localization[cnt_loca:cnt_loca+cnt_loca_k,0]=frame_num
                cnt_loca+=cnt_loca_k
            
        raw_frame=self.stack[0,:,:]*1.
        time.sleep(1.0)
        del self.stack
        time.sleep(5.0)
        if np.shape(results_localization)[0]==0:
            print("WARNING: NO PSF DETECTED IN DATA")
        
        # Export
        if self.generate_localization_image==True and np.shape(results_localization)[0]>0:
            image_loca=self.generate_images(self.limits,results_localization[:,[2,3]])
            skimage.io.imsave(self.filename+"_localization_image.tif",image_loca)
        results_localization[:,2:-1]=self.conversion_output(results_localization[:,2:-1])
        np.save(self.filename+"_localization_results.npy",results_localization)
        self.Save_parameters_final()
        
        print('')
        print('Done in '+str(int(np.ceil(time.time()-self.tref)))+' s')
        print('')
        
Localization_workflow=Localization_workflow()







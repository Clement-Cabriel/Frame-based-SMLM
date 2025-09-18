# Frame-based-SMLM
Localization algorithm for Single-Molecule Localization Microscopy

* AUTHOR:
    Clément Cabriel,
    Institut Langevin, ESPCI Paris / CNRS,
    clement.cabriel@espci.fr , cabriel.clement@gmail.com,
    ORCID: 0000-0002-0316-0312,
    Github: https://github.com/Clement-Cabriel.
If you use the code in a publication, please include a reference to the Github repository (https://github.com/Clement-Cabriel/Frame-based-SMLM). If you would like to use the code for commercial purposes, please contact me
 
* VERSION INFORMATION:
Created on November 23rd 2020, last updated on May 16th 2024.
Tested with Python 3.8.3 and Python 3.8.8
A compatible environment file (SMLM_base_environment.yml) is provided along with the localization code.
 
* CODE DESCRIPTION:
This codes consists of several steps:
  - The image stack is imported 
  - The PSFs are detected on each frame through a wavelet filtering
  - Each PSF is localized by Gaussian fitting
  - The output data (localization list and SMLM image) are exported
Note: this code does not perform drift correction. If it is required, consider using a post-processing code taking the localization list as input
To use the code, set the parametes in the section below, then run the script.

* INPUT FORMATS:
.tif image stack with columns in the following order: x,y,t. If the input stack has a different order, consider modifying the 'import_image_stack' function

* OUTPUT FORMATS:
  - "_localization_image.tif": 2D SMLM image (each molecule counts as 1 in the corresponding pixel)
  - "_localization_results.npy": file containing the localized coordinates. Each row (axis 0) is one molecule, and the different columns (axis 1) contain the different coordinates:
    - Column 0: frame number
    - Column 1: molecule ID
    - Column 2: y (in optical pixels)
    - Column 3: x (in optical pixels)
    - Column 4: PSF y width (in optical pixels or in nm)
    - Column 5: PSF x width (in optical pixels or in nm)
    - Column 6: amplitude of the fitted Gaussian
    - Column 7: background (i.e. offset of the fitted Gaussian)
    - Column 8: fitting error (defined as err=1-R^2, i.e. a good localization is close to 0, and a bad localization is close to 1)
    - Column 9: photon count (i.e. number of photons intedrated over the whole fitting area, background subtracted)
  
    for instance, the element [540,5] of the array is the PSF x witdh of the 540th molecule
   

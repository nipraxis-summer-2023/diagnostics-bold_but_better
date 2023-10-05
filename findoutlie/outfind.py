""" Module with routines for finding outliers
"""

from pathlib import Path
import nibabel as nib
import numpy as np

# import sys
# Put the detect_outliers directory on the Python path.
# PACKAGE_DIR = Path(__file__).parent
# sys.path.append(str(PACKAGE_DIR))

from .detectors import z_score_detector, dvars_detector


def load_fmri_data(fname, dim='4D'):
    """ Loads 4D data from file.
    
    Parametes
    ---------
    fname: string
        name of nibable fMRI image file

    Returns
    --------
    data: 
    4D vector with image data
    """
    img = nib.load(fname)  # Loads the file
    data = img.get_fdata()  # Gets the data from the file

    if dim == '4D':
        return data
    elif dim == '2D':
        num_voxels = np.prod(data.shape[:-1])
        return np.reshape(data, (num_voxels, data.shape[-1]))


def find_outliers(data_directory):
    """ Return filenames and outlier indices for images in `data_directory`.

    Parameters
    ----------
    data_directory : str
        Directory containing containing images.

    Returns
    -------
    outlier_dict : dict
        Dictionary with keys being filenames and values being lists of outliers
        for filename.
    """

    image_fnames = Path(data_directory).glob('**/sub-*.nii.gz')
    outlier_dict = {}
    for fname in image_fnames:
        data = load_fmri_data(fname)  # Gets the 4D data from the file
        
        # outliers = z_score_detector(data)
        outliers = dvars_detector(data)
        outlier_dict[fname] = outliers
    return outlier_dict

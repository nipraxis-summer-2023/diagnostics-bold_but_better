""" Module with routines for finding outliers
"""

from pathlib import Path
import nibabel as nib
import sys

# Put the detect_outliers directory on the Python path.
PACKAGE_DIR = Path(__file__).parent
sys.path.append(str(PACKAGE_DIR))
import detect_outliers as do


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
        img = nib.load(fname)  # Loads the file
        data = img.get_fdata()  # Gets the data from the file
        
        outliers = do.detect_outliers(data)
        outlier_dict[fname] = outliers
    return outlier_dict

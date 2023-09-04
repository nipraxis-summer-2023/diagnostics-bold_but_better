import numpy as np
import sys
import nipraxis as npx
import nibabel as nib

# useful functions when detecting outliers in fMRI data


def vol_mean(data):
    """
    Calculates the mean of each volume in the 4D data
    
    Parameters: a nibabel image
    data: a nibable image (nibabel.nifti1.Nifti1Image), a 4D fMRI data with volumes over time
    --------
    Returns: list
    a list of means, each item being mean per volume across time
    """
    return [
        np.mean(data[..., vol])
        for vol in range(data.shape[-1])
    ]

def vol_std(data):
    """
    Calculates the standard deviation of each volume in the 4D data

    Parameters: nibable image
    data: a nibable image (nibabel.nifti1.Nifti1Image), a 4D fMRI data with volumes over time
    -------
    Returns: list
    A list of standard deviations, each itema being std per volume across time
    """
    return [
        np.std(data[..., vol])
        for vol in range(data.shape[-1])
    ]

def detect_outliers(data, n_std=2):
    """
    Detects outliers in the 4D data and returns a list of indices of outlier volumes

    Parameters: nibable image, int
    data: a nibable image (nibabel.nifti1.Nifti1Image), a 4D fMRI data with volumes over time
    n_std: number of standard deviations away from mean for a volue to be classified as an outlier; default 2
    -------
    Returns: numpy array
    A list of indices of outlier volumes in the data (classifed as > n_std from the mean)
    """
    means = vol_mean(data)
    mean_means = np.mean(means)
    std_means = np.std(means)
    threshold = n_std * std_means
    diff = [
        item - mean_means
        for item in means
    ]
    outliers = np.asarray(np.abs(diff) > threshold).nonzero()[0]

    return outliers


def remove_outliers(data, outliers):
    """
    Removes the outlier volumes in the 4D data and returns data without outliers
    
    Parameters: nibabel image, numpy array
    data: a nibable image (nibabel.nifti1.Nifti1Image), a 4D 
    outliers: A list of indices of outlier volumes in the data
    ------
    Returns: nibabel image data (numpy.memmap, memory map)
    A 4D fMRI data volumes over time, without outlier volumes

    """
    # data is a 4D list, with [x, y, z, volume]
    return [
        [
            [
                [volume for idx, volume in enumerate(z_space) if idx not in outliers] 
                for z_space in y_space
            ]
            for y_space in x_space
        ] 
        for x_space in data
    ]


def main(file_path):
    """
    This function (main) is called when this file run as a script.

    Parameters: str
    A path to nibable image file
    Returns: numpy array
    A list of indices of outlier volumes in the data   
    """
    data_fname = npx.fetch_file(file_path) # Fetches the file
    img = nib.load(data_fname) # Loads the file
    data = img.get_fdata() # Gets the data from the image file

    outliers = detect_outliers(data)
    return outliers
    # return remove_outliers(data, outliers)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 detect_outliers.py <path_to_data>")
        sys.exit(1)
    else:
        main(sys.argv[1])

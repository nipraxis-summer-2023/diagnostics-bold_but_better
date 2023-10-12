""" Utilities for detecting outliers

These functions take a vector of values, and return a boolean vector of the
same length as the input, where True indicates the corresponding value is an
outlier.

The outlier detection routines will likely be adapted to the specific measure
that is being worked on.  So, some detector functions will work on values > 0,
other on normally distributed values etc.  The routines should check that their
requirements are met and raise an error otherwise.
"""

# Any imports you need
import numpy as np

def iqr_detector(data, iqr_factor=1.5, spatial_threshold=0.05):
    """
    Identify outliers along the 4th dimension (time) using the IQR method.
    
    Parameters:
        data (ndarray): 4D fMRI data with shape (x, y, z, t).
        iqr_factor (float): Scaling factor for IQR. Default is 1.5.
        spatial_threshold (float): Proportion of voxels that must agree for a time point to be an outlier.
        
    Returns:
        outlier_time_indices (ndarray): Indices in the 4th dimension where outliers occur.
    """

    # Flatten the first three dimensions to focus on the 4th (time)
    reshaped_data = data.reshape(-1, data.shape[-1])

    # Calculate Q1, Q3 and IQR
    Q1 = np.percentile(reshaped_data, 25, axis=-1)
    Q3 = np.percentile(reshaped_data, 75, axis=-1)
    IQR = Q3 - Q1

    # Compute the outlier bounds
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR

    # Detect outliers in the 4th dimension
    outliers = (reshaped_data < lower_bound[:, np.newaxis]) | (
        reshaped_data > upper_bound[:, np.newaxis])

    # Calculate the proportion of outliers for each time point
    outlier_proportion = np.mean(outliers, axis=0)

    # Find time indices where the proportion of outliers exceeds the spatial threshold
    outlier_time_indices = np.where(outlier_proportion > spatial_threshold)[0]

    return outlier_time_indices


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


def z_score_detector(img_data, n_std=2):
    """
    Detects outliers in the 4D data and returns a list of indices of outlier volumes

    Parameters: nibable image data
    img_data: a nibable image data (nibabel.nifti1.Nifti1Image), a 4D fMRI data with volumes over time
    n_std: number of standard deviations away from mean for a volue to be classified as an outlier; default 2
    -------
    Returns: numpy array
    A list of indices of outlier volumes in the data (classifed as > n_std from the mean)
    """
    if img_data.size == 0:
        return np.array([])
    
    means = vol_mean(img_data)
    mean_means = np.mean(means)
    std_means = np.std(means)
    if std_means == 0: # avoid division by zero
        return np.array([])

    # Calculate Z-scores
    z_scores = (means - mean_means) / std_means

    # Find outliers using Z-scores
    outliers = np.asarray(np.abs(z_scores) > n_std).nonzero()[0]

    return outliers

def dvars(img_data):
    """ Calculate dvars metric on Nibabel image `img`

    The dvars calculation between two volumes is defined as the square root of
    (the mean of the (voxel differences squared)).
    DVARS is essentially like a "spatial RMS", not to be confused with RMS Titanic,
    which also had an impact but not the one we are looking for here. RMS
    is useful mathematically in getting one number for comparison.

    Parameters
    ----------
    img_data : numpy 4D array
        nibabel image data, 4D vector

    Returns
    -------
    dvars : numpy array
        1D array of dvars in `img_data`.
    """

    if img_data.size == 0:
        return np.array([])

    vol_diff = np.diff(img_data, axis=-1)
    return np.sqrt(np.mean(vol_diff ** 2, axis=(0, 1, 2)))
    

def dvars_detector(img_data, z_value=1.96):
    """ Get outliers in Nibabel image `img` based on DVARs calcuation
    
    Parameters
    ----------
    img_data : nibabel image data, 4D vector
    z_value: z value threshold for outlier, default 2

    Returns
    -------
    outlier_volumes : 1D array
        indices of dvars outliers in `img_data`.
    """

    if img_data.size == 0:
        return np.array([])

    dvals = dvars(img_data)
    mean, std = np.mean(dvals), np.std(dvals)
    dynamic_threshold = mean + z_value * std

    dvars_outliers = np.where(dvals > dynamic_threshold)[0]
    outlier_volumes = np.unique(
        np.concatenate([dvars_outliers, dvars_outliers + 1])) # adding one, as outlier could be after

    outlier_volumes = outlier_volumes[outlier_volumes < img_data.shape[3]] # make sure we don't go out of bounds by adding one

    return outlier_volumes


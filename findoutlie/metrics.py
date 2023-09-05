""" Scan outlier metrics
"""

# Any imports you need
import numpy as np

def dvars(img):
    """ Calculate dvars metric on Nibabel image `img`

    The dvars calculation between two volumes is defined as the square root of
    (the mean of the (voxel differences squared)).
    DVARS is essentially like a "spatial RMS", not to be confused with RMS Titanic,
    which also had an impact but not the one we are looking for here. RMS
    is useful mathematically in getting one number for comparison.

    Parameters
    ----------
    img : nibabel image

    Returns
    -------
    dvals : 1D array
        One-dimensional array with n-1 elements, where n is the number of
        volumes in `img`.
    """

    # Hint: remember 'axis='.  For example:
    # In [2]: arr = np.array([[2, 3, 4], [5, 6, 7]])
    # In [3]: np.mean(arr, axis=1)
    # Out[2]: array([3., 6.])
    #
    # You may be be able to solve this in four lines, without a loop.
    # But solve it any way you can.

    data = img.get_fdata()  # get the data array for the image
    vol_diff = np.diff(data, axis=-1) #element wise difference along tha last axis, i.e. volume differences
    dvars = np.sqrt(np.mean(vol_diff ** 2, axis=(0, 1, 2))) # spatial RMS list. Note mean is for first three axis (volume)

    ## another solution is using a for loop:
    # dvars = [] # empty list to store the DVAR number per volume
    # prev_vol = data[...,0] # store first volume
    # for vol in range(1, data.shape[-1]): # loop over nr of volumes in the data, starting with the 2nd volume
    #     this_vol = data[...,vol] # current volume
    #     vol_diff = this_vol - prev_vol # diff between this and previous volume
    #     dvar_val = np.sqrt(np.mean(vol_diff ** 2)) # spatial RMS
    #     dvars.append(dvar_val) # add DVAR to DVAR list
    #     prev_vol = this_vol # set previous volume as current vol for next iteration

    return dvars

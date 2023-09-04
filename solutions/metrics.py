""" Scan outlier metrics
"""

# Any imports you need
# LAB(begin solution)
import numpy as np
# LAB(replace solution)
# +++your code here+++
# LAB(end solution)


def dvars(img):
    """ Calculate dvars metric on Nibabel image `img`

    The dvars calculation between two volumes is defined as the square root of
    (the mean of the (voxel differences squared)).

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
    # LAB(begin solution)
    data = img.get_fdata()
    vx_by_time = np.reshape(data, (-1, data.shape[-1]))
    time_diffs = np.diff(vx_by_time, axis=1)
    return np.sqrt(np.mean(time_diffs ** 2, axis=0))
    # LAB(replace solution)
    # This is a placeholder, replace it to write your solution.
    raise NotImplementedError('Code up this function')
    # LAB(end solution)

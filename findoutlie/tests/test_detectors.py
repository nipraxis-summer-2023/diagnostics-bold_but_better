import nipraxis as npx
import nibabel as nib
import unittest
""" Test script for detector functions

Run these tests with::

    python3 findoutlie/tests/test_detectors.py

or better, in IPython::

    %run findoutlie/tests/test_detectors.py

or even better, from the terminal::

    pytest findoutlie/tests/test_detectors.py

"""

from pathlib import Path
import sys

# This code not needed if you have already run the pip install step:
#
# python3 -m pip install --user --editable .
#
# from the root directory of the repository (the directory containing the
# ``findoutlie`` directory.
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR))
# print(sys.path)

import numpy as np

# This import needs the directory containing the findoutlie directory
# on the Python path.  See above.
from findoutlie.detectors import iqr_detector, z_score_detector, dvars, dvars_detector


def test_iqr_detector():
    # From: http://www.purplemath.com/modules/boxwhisk3.htm
    example_values = np.array(
        [10.2, 14.1, 14.4, 14.4, 14.4, 14.5, 14.5, 14.6, 14.7, 14.7, 14.7,
         14.9, 15.1, 15.9, 16.4])
    is_outlier = iqr_detector(example_values, 1.5)
    assert np.all(example_values[is_outlier] == [10.2, 15.9, 16.4])
    # Test not-default value for outlier proportion
    is_outlier = iqr_detector(example_values, 0.5)
    assert np.all(example_values[is_outlier] == [10.2, 14.1, 15.1, 15.9, 16.4])


def test_z_score_detector():
    # make random 4D vector with 64x64x30 shaped volumes and 10 time points, in range [0, 1]
    data = np.random.rand(64, 64, 30, 10)
    # make two outlier volumes, [2, 6] with significantly higher/lower means
    data[...,2] += 5
    data[...,6] -= 5
    outliers = z_score_detector(data)
    assert np.all(outliers == np.asarray([2, 6]))
    # test for no outliers
    # No outliers; std normal distribution
    data = np.random.normal(0, 1, (64, 64, 30, 10))
    outliers = z_score_detector(data)
    assert np.all(outliers == np.asarray([]))
    #test_empty_data
    data = np.array([])
    outliers = z_score_detector(data)
    assert np.all(outliers == np.asarray([]))
    # test identical volume (ones)
    data = np.ones((64, 64, 30, 10))
    outliers = z_score_detector(data)
    assert np.all(outliers == np.asarray([]))


TEST_FNAME = npx.fetch_file('ds114_sub009_t2r1.nii')

def test_dvars():
    """
    Test the function dvars for DVARS calculation.
    
    The test asserts that the length of calculated DVARS is one less than the number of time points (TRs) in the 4D image.
    It also validates the dvars calculation against a manual calculation, element-wise.
    """
    img = nib.load(TEST_FNAME)
    img_data = img.get_fdata()
    n_trs = img.shape[-1]
    n_voxels = np.prod(img.shape[:-1])

    dvals = dvars(img_data)
    assert len(dvals) == n_trs - 1

    # Manual calculation
    prev_vol = img_data[..., 0]
    long_dvals = []
    for i in range(1, n_trs):
        this_vol = img_data[..., i]
        diff_vol = this_vol - prev_vol
        long_dvals.append(np.sqrt(np.sum(diff_vol ** 2) / n_voxels))
        prev_vol = this_vol

    assert np.allclose(dvals, long_dvals)


def test_dvars_detector():
    # test empty array
    data = np.array([])
    outliers = dvars_detector(data)
    assert np.all(outliers == np.asarray([]))
    # test_all_zeros
    img_data = np.zeros((64, 64, 30, 10))
    assert np.all(outliers == np.asarray([]))
    #test_normal data without outliers
    img_data = np.random.normal(0, 1, (64, 64, 30, 10))
    assert np.all(outliers == np.asarray([]))
    # test normal data with outliers
    img_data = np.random.normal(0, 1, (64, 64, 30, 10))
    img_data[:, :, :, 3:4] = 100  # Introduce outliers
    assert len(dvars_detector(img_data, z_value=1)) > 0
    # test threshold sensitivity
    img_data = np.random.normal(0, 1, (64, 64, 30, 10))
    img_data[:, :, :, 3:5] = 100  # Introduce outliers
    assert len(dvars_detector(img_data, z_value=0.1)) > \
        len(dvars_detector(img_data, z_value=3))


if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    # File being executed as a script
    test_iqr_detector()
    test_z_score_detector()
    test_dvars_detector()
    print('Tests passed')

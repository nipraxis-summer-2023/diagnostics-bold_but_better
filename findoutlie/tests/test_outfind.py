from findoutlie.outfind import load_fmri_data, get_data_files, load_event_data, hrf, convolved_time_course, find_outliers, remove_outliers, glm, evaluate_outlier_methods
import numpy as np
import unittest
from unittest.mock import patch, Mock
from pathlib import Path
import sys
import nibabel as nib
""" Test script for detector functions

Run these tests with::

    python3 findoutlie/tests/test_detectors.py

or better, in IPython::

    %run findoutlie/tests/test_detectors.py

or even better, from the terminal::

    pytest findoutlie/tests/test_detectors.py

"""

# This code not needed if you have already run the pip install step:
#
# python3 -m pip install --user --editable .
#
# from the root directory of the repository (the directory containing the
# ``findoutlie`` directory.
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR))
# print(sys.path)


def test_load_fmri_data():
    with patch('findoutlie.outfind.nib.load') as mock_load:
        # Mock the return value of nib.load
        mock_img = Mock()
        mock_img.get_fdata.return_value = np.random.rand(5, 5, 5, 5)
        mock_load.return_value = mock_img

        # Test for 4D
        result_4D = load_fmri_data(
            'fake_path', dim='4D')
        assert result_4D.shape == (5, 5, 5, 5), "4D shape mismatch"

        # Test for 2D
        result_2D = load_fmri_data(
            'fake_path', dim='2D')
        assert result_2D.shape == (5*5*5, 5), "2D shape mismatch"

def test_get_data_files():
    pass

def test_load_event_data():
    pass
    
def test_hrf():
    pass

def test_convolved_time_course():
    pass

def test_find_outliers():
    pass

def test_remove_outliers():
    pass

def test_glm():
    pass

def test_evaluate_outlier_methods():
    pass


if __name__ == '__main__':
    test_load_fmri_data()
    print("All tests passed. Celebrate responsibly.")

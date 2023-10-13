from unittest.mock import patch, mock_open
import json
import os
import tempfile
import pytest
from unittest.mock import patch
from findoutlie.outfind import load_fmri_data, get_data_files, load_event_data, hrf, convolved_time_course, find_outliers, remove_outliers, glm, evaluate_outlier_methods, write_educated_guess_to_file
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

@patch('findoutlie.outfind.nib.load')
def test_load_fmri_data(mock_load):
    # Mock the return value of nib.load
    mock_img = Mock()
    mock_img.get_fdata.return_value = np.random.rand(5, 5, 5, 5)
    mock_load.return_value = mock_img

    # Test for 4D
    result_4D = load_fmri_data('fake_path', dim='4D')
    assert result_4D.shape == (5, 5, 5, 5), "4D shape mismatch"

    # Test for 2D
    result_2D = load_fmri_data('fake_path', dim='2D')
    assert result_2D.shape == (5*5*5, 5), "2D shape mismatch"


@patch('findoutlie.outfind.Path')
def test_get_data_files(MockPath):
    # Setup Mock
    mock_image_files = [Mock(), Mock()]
    mock_event_files = [Mock(), Mock()]
    MockPath().glob.side_effect = [mock_image_files, mock_event_files]

    # Test function
    image_fnames, event_fnames = get_data_files('dummy_directory')

    # Verify
    assert image_fnames == mock_image_files
    assert event_fnames == mock_event_files
    MockPath().glob.assert_any_call('**/sub-*.nii.gz')
    MockPath().glob.assert_any_call('**/sub-*.tsv')

@patch('numpy.loadtxt') # patching numpy loadtxt directly
def test_load_event_data(mock_loadtxt):
    # setup Mock to return specific value
    mock_loadtxt.return_value = np.array([[0, 30], [30, 30]])

    # Test function
    event_file = load_event_data('dummy_directory')

    # verify
    assert np.array_equal(event_file, np.array(
        [[0, 30], [30, 30]])), "Event data mismatch"
    

def test_hrf():
    # Test times array
    times = np.linspace(0, 30, 61)

    # Calculate HRF
    hrf_values = hrf(times)

    # Ensure peak is at correct location
    peak_time = times[np.argmax(hrf_values)]
    assert peak_time == 5.0, f"Unexpected peak time: {peak_time}"

    # Ensure undershoot is at correct location
    undershoot_time = times[np.argmin(hrf_values)]
    assert undershoot_time == 12.5, f"Unexpected undershoot time: {undershoot_time}"

    # Make sure the max value is scaled to 0.6
    assert np.max(
        hrf_values) == 0.6, f"Unexpected max value: {np.max(hrf_values)}"


# replace 'your_module' with the module where `hrf` lives
@patch('findoutlie.outfind.hrf')
def test_convolved_time_course(mock_hrf):
    # Given
    # replace this with your expected HRF array
    mock_hrf.return_value = np.array([0.1, 0.2, 0.3])
    event_file = np.array([[0, 30], [30, 30]])
    num_vols = 100

    # When
    result = convolved_time_course(event_file, num_vols)

    # Then
    assert result.shape[0] == num_vols, "Shape mismatch in convolved time course"
    # Add more assertions based on your expected output


def test_find_outliers():
    pass


def test_remove_outliers():
    pass


def test_glm():
    pass


def test_write_educated_guess_to_file():
    temp_dir = Path(tempfile.mkdtemp())
    temp_file = temp_dir / 'educated_guess.txt'
    outlier_dict = {
        'z_score_detector': {'outliers': [1, 2, 3]},
        'iqr_detector': {'outliers': [3, 4, 5]},
        'dvars': {'outliers': [5, 6, 7]}
    }
    file_name = "test_file"

    # Run your function
    write_educated_guess_to_file(outlier_dict, file_name, temp_file)

    # Check if file has been created
    assert temp_file.exists()

    # Read the file content
    content = temp_file.read_text()

    # Assertions for expected text in the file
    assert f"--- File: {file_name} ---" in content
    # No common outliers in all three methods in this test
    assert "Common Outliers in all 3 methods" not in content
    # No common outliers between Z-Score and DIVAR
    assert "Z-Score & DIVAR outliers in common" not in content
    assert "Z-Score & IQR outliers in common" in content  # There is a common outlier: 3
    assert "IQR & DIVAR outliers in common" in content  # There is a common outlier: 5


def test_evaluate_outlier_methods():
    pass


if __name__ == '__main__':
    test_load_fmri_data()
    test_get_data_files()
    test_load_event_data()
    test_hrf()
    test_convolved_time_course()
    test_find_outliers()
    test_remove_outliers()
    test_glm()
    test_write_educated_guess_to_file()
    test_evaluate_outlier_methods()
    print("All tests passed. Celebrate responsibly.")

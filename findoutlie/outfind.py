""" Module with routines for finding outliers
"""

from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
from nipraxis.stimuli import events2neural

# import sys
# Put the detect_outliers directory on the Python path.
# PACKAGE_DIR = Path(__file__).parent
# sys.path.append(str(PACKAGE_DIR))

from .detectors import z_score_detector, dvars_detector


def load_fmri_data(fname, dim='4D'):
    """ Loads 4D data from file.
    
    Parametes
    ---------
    fname: full path to image file
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


def get_data_files(data_directory):
    """ Gets image and event files for all subjects from data directory
   
    Parameters
    -----
    data_directory: string
    path to data directory

    Returns
    -----
    image_fnames: iterator to all image (.gz) files in data_directory
    event_fnames: iterator to all event (.gz) files in data_directory
    """
    image_fnames = Path(data_directory).glob('**/sub-*.nii.gz')
    event_fnames = Path(data_directory).glob('**/sub-*.tsv')

    return image_fnames, event_fnames


def load_event_data(event):
    """ Loads an neural event file for a scan

    Parameters
    -----
    event: full path to event file
    Event file for scan

    Returns: 
    event: string
    content of event file
    """
    # Load the file. Skipping the first row (header) and get only first two columns (onset, duration)
    event_onset_duration = np.loadtxt(
        event, skiprows=1, delimiter='\t', usecols=(0, 1))

    return event_onset_duration

def hrf(neural_time_course):
    """ Return values for HRF at given times """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(neural_time_course, 7)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(neural_time_course, 20)
    # Combine them
    values = peak_values - undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6


def convolved_time_course(event_onset_duration):
    """ creates a convolved Hemodynamic Response Function time course
    
    """
    # Determine the total duration of the experiment
    # adding 1 second at the end
    total_duration = int(np.max(event_onset_duration[:, 0])) + 2 # max onset time + 2 seconds
    # Define the time points for convolution 
    TR = 1  # time between scans (assuming 1 Hz sampling rate)
    time_points = np.arange(0, total_duration, TR)

    # Initialize neural_time_course
    neural_time_course = np.zeros(total_duration)

    # Populate neural_time_course
    for onset, duration in event_onset_duration:
        onset_idx = int(onset)
        offset_idx = int(onset + duration)
        neural_time_course[onset_idx:offset_idx] = 1

    # Generate HRF time course
    hrf_time_course = hrf(time_points)

    # Convolve neural_time_course with HRF
    convolved_time_course = np.convolve(
        neural_time_course, hrf_time_course, mode='full')[:len(time_points)]

    # plt.plot(convolved_time_course)
    # plt.legend()
    # plt.show()

    return convolved_time_course


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

    image_fnames, event_fnames = get_data_files(data_directory) # get data files (iterators)
    outlier_methods = ['z_score_detector', 'iqr_detector', 'DIVAR']

    outlier_dict = {} # empty outlier dictionary
    for fname, event in zip(image_fnames, event_fnames): # loop through all files/events in folders
        data = load_fmri_data(fname)  # Gets the 4D data from the file
        event_onset_duration = load_event_data(event)  # Gets the corresponding event file
        # the HRF time course, the y in our modeling
        convolved = convolved_time_course(event_onset_duration)
        
        outliers = select_best_outlier_method(data)
        outlier_dict[fname] = outliers
        
        break  # adding a break for debugging, print out only first file

    return outlier_dict

def select_best_outlier_method(data):
    # for method in outlier_methods:  # loop through outlier methods, per fname/event

        # run GLM and get errors and F-score before removing outliers
        # remove outliers
        # run GLM and get errors and F-score after removing outliers
    return z_score_detector(data)
 
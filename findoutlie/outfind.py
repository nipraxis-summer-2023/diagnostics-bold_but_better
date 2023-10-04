from scipy.stats import f
from statsmodels.api import OLS, add_constant
from sklearn.metrics import mean_squared_error
from sklearn.covariance import EllipticEnvelope
import numpy.linalg as npl
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

""" Module with routines for finding outliers
"""

# import sys
# Put the detect_outliers directory on the Python path.
# PACKAGE_DIR = Path(__file__).parent
# sys.path.append(str(PACKAGE_DIR))

from .detectors import z_score_detector, iqr_detector

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
    event_file = np.loadtxt(
        event, skiprows=1, delimiter='\t', usecols=(0, 1))

    return event_file


def hrf(times):
    """ Return values for HRF at given times
    
    Parameters
    ------
    times: numpy array

    Returns:
    ------
    HRF function for times
    """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6


def convolved_time_course(event_file, num_vols):
    """ creates a convolved Hemodynamic Response Function model from neuronal model (event)
    
    Parameters:
    -----
    event_onset_and_duration: str
    the event file with onset and duration (two columns)
    num_vols: int
    number of volumes in scan

    Returns:
    convolved_time_course: 
    HRF model for data
    """    
    TR = 3  # time between scans (3 Hz sampling rate), from the nib header file (pixdim)
    onsets_seconds = event_file[:, 0] # from column 1
    durations_seconds = event_file[:, 1]  # from column 2
    amplitude = 1 # we don´t have amplitudes in event file so we set all onset to 1

    onsets_in_scans = onsets_seconds / TR
    tr_divs = 100.0  # finer resolution has 100 steps per TR
    high_res_times = np.arange(0, num_vols, 1 / tr_divs) * TR
    high_res_neural = np.zeros(high_res_times.shape)
    high_res_onset_indices = onsets_in_scans * tr_divs # create indices for high res
    high_res_durations = durations_seconds / TR * tr_divs
    for hr_onset, hr_duration in zip(
            high_res_onset_indices, high_res_durations):
        hr_onset = int(round(hr_onset))  # index - must be int
        hr_duration = int(round(hr_duration))  # makes index - must be int
        high_res_neural[hr_onset:hr_onset + hr_duration] = amplitude

    # show high res 
    # plt.plot(high_res_times, high_res_neural)
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('High resolution neural prediction')
    # plt.show()

    hrf_times = np.arange(0, 24, 1 / tr_divs)
    hrf_at_hr = hrf(hrf_times)
    high_res_hemo = np.convolve(high_res_neural, hrf_at_hr) # convolve high res

    # Drop tail from convolution
    high_res_hemo = high_res_hemo[:len(high_res_neural)]

    # plt.plot(high_res_times, high_res_hemo)
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('High resolution convolved values')
    # plt.show()
    # print(f'len(high_res_times): {len(high_res_times)}')

    # subsample down
    tr_indices = np.arange(num_vols)
    hr_tr_indices = np.round(tr_indices * tr_divs).astype(int)
    tr_hemo = high_res_hemo[hr_tr_indices]

    # tr_times = tr_indices * TR  # times of TR onsets in seconds
    # plt.plot(tr_times, tr_hemo)
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Convolved values at TR onsets')
    # plt.show()
    # downsampled view
    # plt.plot(tr_times[:20], tr_hemo[:20], 'x:')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Convolved values at TR onsets')
    # plt.show()

    return tr_hemo


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

    outlier_dict = {} # empty outlier dictionary
    for fname, event in zip(image_fnames, event_fnames): # loop through all files/events in folders
        data = load_fmri_data(fname)  # Gets the 4D data from the file
        num_vols = data.shape[-1] # number of volumes in scan
        # Gets the corresponding event file
        event_file = load_event_data(event)
        # the HRF time course, the y in our modeling
        convolved = convolved_time_course(event_file, num_vols)
        
        outliers = evaluate_outlier_methods(data, convolved)
        # outliers = z_score_detector(data) # 2DO SETTING THIS TEMPORARY TO RUN CODE
        outlier_dict[fname] = outliers
        
        return outlier_dict  # TEMP adding a BREAK for debugging, ONLY RUN first IMG file. REMOVE

    return outlier_dict


from sklearn.datasets import make_regression

def remove_outliers(data, method):
    if method == 'z_score_detector':
        outliers = z_score_detector(data)
    elif method == 'iqr_detector':
        outliers = iqr_detector(data)
    elif method == 'DIVAR':
        NotImplemented
    else:
        return NotImplemented
    
    filtered_data = np.delete(data, outliers, axis=3)

    return filtered_data, outliers # return data without outliers and outlier indices

def glm(data, factors, c):
    N = data.shape[-1]
    X = np.ones((N, len(factors) + 1))
    X[:, 1:] = np.column_stack(factors) # add each factor as a column in design matrix
    Y = np.reshape(data, (-1, N))  # reshape data to time by voxels
    Y = Y.T  # transpose
    B = npl.pinv(X) @ Y  # y_hat = X@B_hat + error

    # t test
    top_of_t = c @ B
    np.all(top_of_t == B[1, :])
    df_error = N - npl.matrix_rank(X)
    fitted = X @ B
    E = Y - fitted
    sigma_2 = np.sum(E ** 2, axis=0) / df_error
    c_b_cov = c.T @ npl.pinv(X.T @ X) @ c
    t = top_of_t / np.sqrt(sigma_2 * c_b_cov)

    return X, Y, E, t

def evaluate_outlier_methods(data, convolved):
    data = data[...,1:] # knock of first scan
    convolved = convolved[1:]  # knock of first item for consistency

    # glm stuff
    # contrast matrix: Contrast the difference of the slope from 0
    c = np.array([0, 1])
    X, Y, E, t = glm(data, [convolved], c)

    # mask of the voxels within the brain using Otsu’s method
    mean = np.mean(data, axis=-1)
    thresh = threshold_otsu(mean)
    mask = mean > thresh
    # plt.imshow(mask[:, :, 15], cmap='gray')
    # plt.show()

    N = data.shape[-1]
    Y = data[mask].T
    B = npl.pinv(X) @ Y
    fitted = X @ B
    E = Y - fitted
    df_error = N - npl.matrix_rank(X)
    sigma_2 = np.sum(E ** 2, axis=0) / df_error
    # c and c_b_cov are the same as before, but recalculate anyway
    c = np.array([0, 1])
    c_b_cov = c @ npl.pinv(X.T @ X) @ c
    t = c.T @ B / np.sqrt(sigma_2 * c_b_cov)
    # print(f't.shape {t.shape}')
    # t_3d = np.zeros(data.shape[:3])
    # t_3d[mask] = t
    # plt.imshow(t_3d[:, :, 15], cmap='gray')
    # plt.show()

    import scipy.stats as stats
    # calc p value for each t statistic
    t_dist = stats.t(df_error)
    p = 1 - t_dist.cdf(t)
    # print(p.shape)
    # p_3d = np.zeros(data.shape[:3])
    # p_3d[mask] = p
    # plt.imshow(p_3d[:, :, 15], cmap='gray')
    # plt.show()

    # multiple comparision correction, bonferroni
    N = p.shape[0]
    bonferroni_thresh = 0.05 / N
    # print(f'bonferroni_thresh: {bonferroni_thresh}')
    # plt.imshow(p_3d[:, :, 15] < bonferroni_thresh, cmap='gray')
    # plt.show()

    
    methods = ['z_score_detector', 'iqr_detector', 'DIVAR']
    outlier_perf = {}
    for method in methods:
        # Remove outliers
        data_filtered, outliers = remove_outliers(data, method)
        convolved_filtered = np.delete(convolved, outliers)
        X_filtered, y_filtered, E, t = glm(data_filtered, [convolved_filtered], c)

        # Fit GLM on original data
        model_orig = OLS(Y, add_constant(X)).fit()
        rms_orig = np.sqrt(mean_squared_error(Y, model_orig.fittedvalues))

        # Fit GLM on filtered data
        model_filtered = OLS(y_filtered, add_constant(X_filtered)).fit()
        rms_filtered = np.sqrt(mean_squared_error(
            y_filtered, model_filtered.fittedvalues))

        # Perform F-test between original and filtered models
        f_stat, p_value = f(model_orig.df_resid, model_filtered.df_resid,
                            model_orig.ssr, model_filtered.ssr)

        # Report metrics
        print(f"Method: {method}")
        print(f"RMS before: {rms_orig}")
        print(f"RMS after: {rms_filtered}")
        print(f"F-test statistic: {f_stat}")
        print(f"P-value: {p_value}")
        outlier_perf[method] = {"RMS before": rms_orig, "RMS after": rms_filtered,
                                "F-test statistic": f_stat, "P-value": p_value, "outliers": outliers}


    best_method = min(outlier_perf.keys(),
                  key=lambda x: outlier_perf[x]['RMS after'])
    
    return outlier_perf[best_method][outliers]

    # for method in outlier_methods:  # loop through outlier methods, per fname/event

        # run GLM and get errors and F-score before removing outliers
        # remove outliers
        # run GLM and get errors and F-score after removing outliers
    # print out results in a text file (our arguments for selecting best outlier)

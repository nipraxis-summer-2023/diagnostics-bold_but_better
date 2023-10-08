from statsmodels.stats.multitest import multipletests
from scipy import stats
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

from .detectors import z_score_detector, iqr_detector, dvars_detector

def load_fmri_data(fname, dim='4D'):
    """ Loads 4D data from file.
    
    Parametes
    ---------
    fname: string
        full path to fMRIimage file
    dim: str
        shape of returned data, either '4D' or '2D'

    Returns
    --------
    data: numpy array
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
    image_fnames: iterator
        for all image (.gz) files in data_directory
    event_fnames: iterator
        for all event (.gz) files in data_directory
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
    convolved_time_course: numpy array
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


def find_outliers(data_directory, verbose=False):
    """ Return filenames and outlier indices for images in `data_directory`.

    Parameters
    ----------
    data_directory : str
        Directory containing containing images.
    verbose: bool
        display images and logs if True

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
        
        outliers = evaluate_outlier_methods(data, convolved, verbose=verbose)
        outlier_dict[fname] = outliers
        
        # return outlier_dict  # TEMP adding a BREAK for debugging, ONLY RUN first IMG file. REMOVE

    return outlier_dict

def remove_outliers(data, method):
    """remove outliers based on method and return data without outliers
    
    Parameters:
    ------
    data: numpy 4D array
        a 4D fMRI image data
    metohd: string
        name of availabe outlier detection methods: 'z_score_detector', 'iqr_detector', 'DIVAR'

    Returns:
    ------
    filtered_data: numpy 4D array
        a 4D fMRI image data without outliers
    outliers: numpy 1D array
        indicies of outliers in unfiltered data
    """
    if method == 'z_score_detector':
        outliers = z_score_detector(data)
        # print(f'outliers z: {outliers}')
    elif method == 'iqr_detector':
        outliers = iqr_detector(data)
        # print(f'outliers iqr: {outliers}')
    elif method == 'dvars':
        outliers = dvars_detector(data, z_value=1.645) # setting z value at 90% CI
    else:
        return NotImplemented
    
    filtered_data = np.delete(data, outliers, axis=3)
    # print(f'data shape {data.shape}, filtered shape {filtered_data.shape}')

    return filtered_data, outliers # return data without outliers and outlier indices


def glm(data, factors, c, otsu_mask=True, mult_comp='fdr_bh', axes=None, title='', slice=15):
    """General linear model
    
    Parametes:
    ---------
    data: numpy 4D array
        a 4D fMRI image data
    factors: list
        a list of factors with numpy arrays of N shape for the GLM
    c: list
        the control matrix for the GLM
    otsu_mask: Bool
        whether to apply otsu mask filtering on data
    mult_comp: str
        the multiple comparison correction method: 'bonferroni', 'fdr_bh', 'holm' or 'sidak'
    show: bool
        wheter to display plots or notd
    title: str
        supplment title for plots
    slice: int
        nr of slice in volume to plot

    Returns:
    --------
    X: numpy array
        The design matrix
    Y: numpy array
        The predicted value (y_hat)
    E: numpy array
        The errors or residuals (Y - Y_hat)
    t: numpay array
        t values per voxel
    p: numpay array
        p values per voxel
    p_adj: numpay array
        Multiple comparison corrected p-values per voxel
    """
    N = data.shape[-1]
    if otsu_mask:
        mean = np.mean(data, axis=-1)
        thresh = threshold_otsu(mean)
        mask = mean > thresh
        # plt.imshow(mask[:, :, 15], cmap='gray')
        # plt.title("Otsu's mask applied")
        # plt.show()
    else:
        mask = None  # no mask

    if mask is not None:
        Y = np.reshape(data[mask], (-1, N))
    else:
        Y = np.reshape(data, (-1, N))

    Y = Y.T
    X = np.ones((N, len(factors) + 1))
    X[:, 1:] = np.column_stack(factors)
    B = npl.pinv(X) @ Y

    top_of_t = c @ B
    df_error = N - npl.matrix_rank(X)
    fitted = X @ B
    E = Y - fitted
    sigma_2 = np.sum(E ** 2, axis=0) / df_error
    c_b_cov = c.T @ npl.pinv(X.T @ X) @ c

    # catch division by zero
    with np.errstate(divide='ignore', invalid='ignore'): # catch division by zero
        t = np.true_divide(top_of_t, np.sqrt(sigma_2 * c_b_cov))
        t[~np.isfinite(t)] = np.nan  # -inf, inf, NaN
    # print(f't shape: {t.shape}')

    t_3d = np.zeros(data.shape[:3])
    p_3d = np.zeros(data.shape[:3])
    t_dist = stats.t(df_error)
    p = 1 - t_dist.cdf(t)
    # print(f'p shape: {p.shape}')

    if mask is not None:
        t_3d[mask] = t
        p_3d[mask] = p
    else:
        t_3d = np.reshape(t, data.shape[:3])
        p_3d = np.reshape(p, data.shape[:3])
    # print(f't_3d / p_3d shapes: {t_3d.shape} / {p_3d.shape}')
    
    ## multiple comparison correction:
    #   Bonferroni ('bonferroni')
    #   Benjamini-Hochberg's FDR ('fdr_bh')
    #   Holm: 'holm'
    #   Sidak: 'sidak'
    p_flat = p.ravel()
    reject, pvals_corrected, _, _ = multipletests(
        p_flat, alpha=0.05, method=mult_comp.lower())
    p_adj = np.zeros(p_3d.shape)
    if mask is not None:
        p_adj[mask] = pvals_corrected
    else:
        p_adj = np.reshape(pvals_corrected, p_3d.shape)  # reshape p-values into 3D
    # print(f'p_adj shape: {p_adj.shape}')

    # show t scores per voxel
    if axes is not None:
        axes[0].imshow(t_3d[:, :, slice], cmap='gray')
        axes[0].set_title(f'slice of t scores per voxel: {title}')

        # show p-values per voxel
        axes[1].imshow(p_3d[:, :, slice], cmap='gray')
        axes[1].set_title(f'slice of p scores per voxel: {title}')

        # show p-adjusted values per voxel
        axes[2].imshow(p_adj[:, :, slice], cmap='gray')
        axes[2].set_title(
            f'slice of corrected p-values, {mult_comp} : {title}')

        for ax in axes:
            ax.axis('off')
    return X, Y, E, t, p, p_adj


def evaluate_outlier_methods(data, convolved, verbose=False):
    """Run different outlier detction methods and select the best one
    
    Parameters:
    -------
    data: 4D numpy array
        The 4D nibable image data
    convolved: 1D numpy array
        The hemodynamic response model
    verbose: bool
        display images and print logs

    Returns:
    -------
    outliers_best_method: numpy array
        indices of outliers from the best outlier detection method
    """
    methods = ['z_score_detector', 'iqr_detector', 'dvars']
    outlier_perf = {}
    for method in methods:
        # Create a 2x3 subplot grid, if show=True
        # Create a new figure for each method
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        slice = 15 # set which slice to display

        data = data[..., 1:]  # knock of first scan
        convolved = convolved[1:]  # knock of first item for consistency
        # contrast matrix: Contrast the difference of the slope from 0
        c = np.array([0, 1])
        X, Y, E, t, p, p_adj = glm(
            data, [convolved], c, otsu_mask=True, mult_comp='fdr_bh', axes=axes[0, :], slice=slice)
        # print(f'X {X.shape}, Y {Y.shape}, E {E.shape}, t {t.shape}, p {p.shape}, p_adj {p_adj.shape}')

        # Remove outliers
        data_filtered, outliers = remove_outliers(data, method)
        convolved_filtered = np.delete(convolved, outliers)
        X_filtered, y_filtered, E_filt, t_filt, p_filt, p_adj_filt = glm(
            data_filtered, [convolved_filtered], c, otsu_mask=True, mult_comp='fdr_bh', axes=axes[1, :], title='Filtered data', slice=slice)

        # compare MRSS between volumes, original and filtered
        MRSS = []
        for vol, resid in [(X, E), (X_filtered, E_filt)]:
        # Residual sum of squares
            RSS = np.sum(resid ** 2)
            # Degrees of freedom: n - no independent columns in X
            df = vol.shape[0] - npl.matrix_rank(vol)
            # Mean residual sum of squares
            MRSS.append(RSS / df)
        drop = np.around(1 - MRSS[1]/MRSS[0], 4) * 100
        # print(f'Drop in MRSS: {(1 - MRSS[1]/MRSS[0]) * 100:.2f}%')
        outlier_perf[method] = {'MRSS before': {
            MRSS[0]}, 'MRSS after': MRSS[1], 'drop (%)': drop, 'outliers': outliers}

        if verbose:
            print(
                f'\nApplying outlier detection method: \033[1m{method}\033[0m')
            escaped_method = method.replace('_', '\\_')
            fig.suptitle(f't, p and p_adj values: $\\bf{{{escaped_method}}}$ method gives a {drop}% drop in MRSS.\n\
            Original on top and filtered below for slice nr: {slice}')
            plt.show()
            plt.close(fig)
            print(f'\tMRSS for dataset before removing outliers: {np.around(MRSS[0], 4)}\n \
                MRSS for dataset after removing outliers: {np.around(MRSS[1], 4)},\n \
                a reduction of \033[1m{drop}%\033[0m\n\n')

    # print(outlier_perf)
    best_method = max(outlier_perf.keys(),
                      key=lambda x: outlier_perf[x]['drop (%)'])
    
    outliers_best_method = outlier_perf[best_method]['outliers']

    if verbose:
        print(f'\033[1m{best_method}\033[0m gives the biggest reduction on MRSS and is therefore selected\n\
            with file name and indices of outliers per volume as follows:\n')

    return outliers_best_method


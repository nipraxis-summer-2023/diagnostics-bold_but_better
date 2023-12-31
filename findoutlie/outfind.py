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


def find_outliers(data_directory, verbose=False, show=False, conservative=False):
    """ Return filenames and outlier indices for images in `data_directory`.

    Parameters
    ----------
    data_directory : str
        Directory containing containing images.
    verbose: bool
        print logs
    show: bool
        display images
    conservative: bool
        combine outliers from all methods if True

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

        # Check for empty data
        if data.size == 0:
            print(f"Warning: Empty data for file {fname}. Skipping.")
            continue  # Skip to next iteration

        num_vols = data.shape[-1] # number of volumes in scan
        # Gets the corresponding event file
        event_file = load_event_data(event)
        # the HRF time course, the y in our modeling
        convolved = convolved_time_course(event_file, num_vols)
        
        outliers = evaluate_outlier_methods(
            fname, data, convolved, verbose=verbose, show=show, conservative=conservative)
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
    outliers = np.array([])
    if method == 'z_score_detector':
        outliers = z_score_detector(data)
        # print(f'outliers z: {outliers}')
    elif method == 'iqr_detector':
        # setting iqr factor to 1.5 spatial threshold to 2%
        outliers = iqr_detector(data, iqr_factor=1.5, spatial_threshold=0.02)
        # print(f'outliers iqr: {outliers}')
    elif method == 'dvars':
        # setting z value at 90% CI and spatial threshold to 2%
        outliers = dvars_detector(data, z_value=1.645)
        # print(f'outliers divars: {outliers}')
    else:
        return NotImplemented
    
    filtered_data = np.delete(data, outliers, axis=3)
    assert len(outliers) + filtered_data.shape[-1] == data.shape[-1]

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
    if data.size == 0:
        raise ValueError("The input data array is empty. Cannot proceed.")
    N = data.shape[-1]
    if otsu_mask:
        mean = np.mean(data, axis=-1)
        # Check for NaN or infinite values in mean
        if np.isnan(mean).any() or np.isinf(mean).any():
            print(
                "Warning: NaN or infinite values detected in mean. Replacing with zeros.")
            mean = np.nan_to_num(mean)

        thresh = threshold_otsu(mean)
        mask = mean > thresh
        # plt.imshow(mask[:, :, 15], cmap='gray')
        # plt.title("Otsu's mask applied")
        # plt.show()
    else:
        mask = None  # no mask

    if mask is not None:
        if np.any(mask):  # Make sure mask is not empty
            Y = np.reshape(data[mask], (-1, N))
        else:
            print("Warning: The mask didn't capture any data. Proceeding without masking.")
            Y = np.reshape(data, (-1, N))
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


def write_educated_guess_to_file(outlier_dict, file_name, file_path=Path('educated_guess.txt')):
    """
    Append educated guesses about outliers to a text file, based on outlier indices from different detection methods.
    
    Parameters:
    -----------
    outlier_dict : dict
        Dictionary containing outliers and some statistics, indexed by the detection method used.
    file_name : str
        The name of the file being analysed, to be written in the text file for reference.
    file_path : Path
        The Path object pointing to the output text file.
        
    Returns:
    --------
    None
    
    Side Effect:
    ------------
    Appends educated guesses to 'educated_guess.txt' based on the outlier indices provided in `outlier_dict`.
    """
    with open(file_path, 'a') as f:
        f.write(f"--- File: {file_name} ---\n\n")

        common_outliers = set(outlier_dict['z_score_detector']['outliers']) & \
            set(outlier_dict['iqr_detector']['outliers']) & \
            set(outlier_dict['dvars']['outliers'])

        if common_outliers:
            f.write("Common Outliers in all 3 methods: Most likely significant issues, such as extreme motion or hardware failure.\n\n")
            return

        z_and_d = set(outlier_dict['z_score_detector']['outliers']) & set(
            outlier_dict['dvars']['outliers'])
        z_and_i = set(outlier_dict['z_score_detector']['outliers']) & set(
            outlier_dict['iqr_detector']['outliers'])
        i_and_d = set(outlier_dict['iqr_detector']['outliers']) & set(
            outlier_dict['dvars']['outliers'])

        if z_and_d:
            f.write(
                "Z-Score & DIVAR outliers in common: Likely due to widespread changes; could be task-related or head motion.\n")
        if z_and_i:
            f.write(
                "Z-Score & IQR outliers in common: Localised but significant spikes; possible artifact or physiological change.\n")
        if i_and_d:
            f.write(
                "IQR & DIVAR outliers in common: Unusual data points, could be related to complex motion or equipment issues.\n")

        f.write("\n")


def evaluate_outlier_methods(fname, data, convolved, verbose=False, show=False, conservative=False):
    """Run different outlier detction methods and select the best one
    
    Parameters:
    -------
    fname: str
        name of image file
    data: 4D numpy array
        The 4D nibable image data
    convolved: 1D numpy array
        The hemodynamic response model
    verbose: bool
        print logs
    show: bool
        display images
    conservative: bool
        combine outliers from all methods

    Returns:
    -------
    outliers_best_method: numpy array
        indices of outliers from the best outlier detection method
    """
    methods = ['z_score_detector', 'iqr_detector', 'dvars']
    outlier_perf = {}
    data = data[..., 1:]  # knock of first scan
    convolved = convolved[1:]  # knock of first item for consistency
    if verbose:
        print(f'\n------{fname}------')
    for method in methods:
        # Create a 2x3 subplot grid, if show=True
        # Create a new figure for each method
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        slice = 15 # set which slice to display

        # data = data[..., 1:]  # knock of first scan
        # convolved = convolved[1:]  # knock of first item for consistency
        # contrast matrix: Contrast the difference of the slope from 0
        c = np.array([0, 1])
        X, Y, E, t, p, p_adj = glm(
            data, [convolved], c, otsu_mask=True, mult_comp='fdr_bh', axes=axes[0, :], slice=slice)
        # print(f'X {X.shape}, Y {Y.shape}, E {E.shape}, t {t.shape}, p {p.shape}, p_adj {p_adj.shape}')

        # Remove outliers
        data_filtered, outliers = remove_outliers(data, method)
        convolved_filtered = np.delete(convolved, outliers)

        # Check for empty filtered data
        if data_filtered.size == 0 or convolved_filtered.size == 0:
            print(
                f"Warning: {method} considered all data points to be outliers for file: {fname}. \nSkipping method...")
            outlier_perf[method] = {'MRSS before': {0}, 'MRSS after': 0, 'drop (%)': 0, 'outliers': []}
            continue  # Skip to the next iteration

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
        drop = round(np.around(1 - MRSS[1]/MRSS[0], 4) * 100, 4)
        # print(f'Drop in MRSS: {(1 - MRSS[1]/MRSS[0]) * 100:.2f}%')
        outlier_perf[method] = {'MRSS before': {
            MRSS[0]}, 'MRSS after': MRSS[1], 'drop (%)': drop, 'outliers': outliers}

        if verbose:
            print(f'Applying outlier detection method: \033[1m{method}\033[0m')
            print(f'\tMRSS for dataset before/after removing outliers: {np.around(MRSS[0], 4)} / {np.around(MRSS[1], 4)},\n \
            a reduction of \033[1m{drop}%\033[0m. \n\
            Indices of outliers: {outlier_perf[method]["outliers"]}')
        if show:
            escaped_method = method.replace('_', '\\_')
            fig.suptitle(f'File: {fname}\n \
            t, p and p_adj values: $\\bf{{{escaped_method}}}$ method gives a {drop}% drop in MRSS.\n\
            Original on top and filtered below for slice nr: {slice}')
            plt.show()
            plt.close(fig)
        
        plt.close(fig)
    
    write_educated_guess_to_file(
        outlier_perf, fname)  # write to text file
    
    best_method = max(outlier_perf.keys(),
                      key=lambda x: outlier_perf[x]['drop (%)']) # find best method, one with max reductin on MRSS
    outliers_best_method = outlier_perf[best_method]['outliers']

    if verbose:
        print(f'\nOf these outlier detection methods, the biggest reduction on MRSS comes from \033[1m{best_method}\033[0m method\n')

    if not conservative:
        return outliers_best_method
    else:
        ## Let's combine outliers from all three methods
        common_outliers = set(outlier_perf['z_score_detector']['outliers']) | set(
            outlier_perf['iqr_detector']['outliers']) | set(outlier_perf['dvars']['outliers'])
        common_outliers_all = np.asarray([int(x) for x in set(common_outliers)])
        common_outliers_all.sort()

        # Remove all common outliers
        data_filt_all = np.delete(data, common_outliers_all, axis=3)
        convolved_filt_all = np.delete(convolved, common_outliers_all)
        X_filt_all, y_filt_all, E_filt_all, t_filt_all, p_filt_all, p_adj_filt_all = glm(
            data_filt_all, [convolved_filt_all], c, otsu_mask=True, mult_comp='fdr_bh', axes=axes[1, :], title='All methods filtered', slice=slice)
        RSS_all = np.sum(E_filt_all ** 2)
        # Degrees of freedom: n - no independent columns in X
        df_all = X_filt_all.shape[0] - npl.matrix_rank(X_filt_all)
        # Mean residual sum of squares
        MRSS_all = RSS_all / df_all
        drop_all = round(np.around(1 - MRSS_all/MRSS[0], 4) * 100, 4)
        ##

        if verbose:
            print(
                f'###\nCombining outliers from all methods:')
            print(f'MRSS for dataset before / after removing outliers: {np.around(MRSS[0], 4)} / {MRSS_all},\n \
        a reduction of \033[1m{drop_all}%\033[0m.\n\
 All outliers: {common_outliers_all}\n\
### \n')

        return common_outliers_all


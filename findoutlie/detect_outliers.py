import numpy as np
import sys
import nipraxis as npx
import nibabel as nib

# useful functions when detecting outliers in fMRI data


def vol_mean(data):
    """
    Calculates the mean of the volume of the data
    """
    return [
        np.mean(data[..., vol])
        for vol in range(data.shape[-1])
    ]

def vol_std(data):
    """
    Calculates the standard deviation of the volume of the data
    """
    return [
        np.std(data[..., vol])
        for vol in range(data.shape[-1])
    ]

def detect_outliers(data, n_std=2):
    """
    Detects outliers in the data and returns a list of indices of the outliers
    """
    means = vol_mean(data)
    mean_means = np.mean(means)
    std_means = np.std(means)
    threshold = n_std * std_means
    diff = [
        item - mean_means
        for item in means
    ]
    outliers = np.asarray(np.abs(diff) > threshold).nonzero()[0]

    return outliers


def remove_outliers(data, outliers):
    """
    Removes the outliers from the data and returns the new data
    """
    # data is a 4D list, with [x, y, z, volume]
    return [
        [
            [
                [volume for idx, volume in enumerate(z_space) if idx not in outliers] 
                for z_space in y_space
            ]
            for y_space in x_space
        ] 
        for x_space in data
    ]


def main(file_path):
    """
    Main function
    """
    data_fname = npx.fetch_file(file_path) # Fetches the file
    img = nib.load(data_fname) # Loads the file
    data = img.get_fdata() # Gets the data from the file

    outliers = detect_outliers(data)
    return remove_outliers(data, outliers)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 detect_outliers.py <path_to_data>")
        sys.exit(1)
    else:
        main(sys.argv[1])

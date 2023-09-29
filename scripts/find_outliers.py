""" Python script to find outliers

Run as:

    python3 scripts/find_outliers.py data
"""

from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# add necessary directories to python's search path at runtime
import sys
PACKAGE_DIR = Path(__file__).parent / '..'
sys.path.append(str(PACKAGE_DIR))


from findoutlie import find_outliers

def outliers(data_directory):
    """ Creates a dictionary with file name and incidies of outlier 
     volumes per file (scan) in the data directory

    Parameters:
    -------
    data_directory: string
    path to data directory

    Returns:
    ------
    outlier_dict: dict
    dictionary with indicies of outliers per file
    """
    outlier_dict = find_outliers(data_directory)
    
    return outlier_dict


def print_outliers(outlier_dict):
    """ Prints file name and incidies of outlier volumes per scan in data directory

    Parameters:
    -------
    data_directory: string
    path to data directory

    Returns:
    ------
    nothing
    """
    for fname, outliers in outlier_dict.items():
        if len(outliers) == 0:
            continue
        outlier_strs = []
        for out_ind in outliers:
            outlier_strs.append(str(out_ind))
        print(', '.join([str(fname)] + outlier_strs))


def get_parser():
    parser = ArgumentParser(description=__doc__,  # Usage from docstring
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('data_directory',
                        help='Directory containing data')
    return parser


def main():
    # This function (main) called when this file run as a script.
    #
    # Get the data directory from the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    # Call function to find outliers.
    print_outliers(args.data_directory)


if __name__ == '__main__':
    # Python is running this file as a script, not importing it.
    main()

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

def get_outliers(data_directory, verbose, show, conservative):
    """ Creates a dictionary with file name and incidies of outlier 
     volumes per file (scan) in the data directory

    Parameters:
    -------
    data_directory: string
        path to data directory
    verbose: bool
        print logs
    show: bool
        display images
    conservative: bool
        combine outliers from all methods if True

    Returns:
    ------
    outlier_dict: dict
    dictionary with indicies of outliers per file
    """
    outlier_dict = find_outliers(data_directory, verbose, show, conservative)

    return outlier_dict


def print_outliers(data_directory, verbose, show, conservative):
    """ Prints file name and incidies of outlier volumes per scan in data directory

    Parameters:
    -------
    data_directory: string
        path to data directory
    verbose: bool
        print logs
    show: bool
        display images
    conservative: bool
        combine outliers from all methods if True

    Returns:
    ------
    nothing
    """
    outlier_dict = get_outliers(data_directory, verbose, show, conservative)
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
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase output verbosity.")
    parser.add_argument("-s", "--show", action="store_true",
                        help="Show images.")
    parser.add_argument("-c", "--conservative", action="store_true",
                        help="Combine outliers from all methods.")

    return parser


def main():
    # This function (main) called when this file run as a script.
    #
    # Get the data directory from the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    verbose = args.verbose
    show = args.show
    conservative = args.conservative
    # Call function to find outliers.
    print_outliers(args.data_directory, verbose, show, conservative)


if __name__ == '__main__':
    # Python is running this file as a script, not importing it.
    main()

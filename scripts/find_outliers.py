""" Python script to find outliers

Run as:

    python3 scripts/find_outliers.py data
"""

from pathlib import Path
import sys

from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Put the findoutlie directory on the Python path.
PACKAGE_DIR = Path(__file__).parent / '..'
sys.path.append(str(PACKAGE_DIR))

from findoutlie import outfind


def print_outliers(data_directory):
    outlier_dict = outfind.find_outliers(data_directory)
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

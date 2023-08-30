""" Python script to validate data

Run as:

    python3 scripts/validate_data.py data
"""

from pathlib import Path
import sys
import hashlib


def file_hash(filename):
    """ Get byte contents of file `filename`, return SHA1 hash

    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    hash : str
        SHA1 hexadecimal hash string for contents of `filename`.
    """
    # Open the file, read contents as bytes.
    file_contents = Path(filename).read_bytes()
    # Calculate, return SHA1 has on the bytes from the file.
    
    return hashlib.sha1(file_contents).hexdigest()


def validate_data(data_directory):
    """ Read ``data_hashes.txt`` file in `data_directory`, check hashes

    Parameters
    ----------
    data_directory : str
        Directory containing data and ``data_hashes.txt`` file.

    Returns
    -------
    None

    Raises
    ------
    ValueError:
        If hash value for any file is different from hash value recorded in
        ``data_hashes.txt`` file.
    """
    # Read lines from ``data_hashes.txt`` file.
    hashes_pth = Path(data_directory) / 'group-00' / 'hash_list.txt'
    # Split into SHA1 hash and filename
    lines = hashes_pth.read_text().splitlines()
    # Calculate actual hash for given filename.
    for line in lines:
        exp_hash, fname = line.split() # Split each line into expected_hash and filename
        calc_hash = file_hash(Path(data_directory) / fname)
        # If hash for filename is not the same as the one in the file, raise ValueError
        if calc_hash != exp_hash:
            raise ValueError(f'Hash for {fname} is {calc_hash}, which does not match {exp_hash}')


def main():
    # This function (main) called when this file run as a script.
    #
    # Get the data directory from the command line arguments
    if len(sys.argv) < 2:
        raise RuntimeError("Please give data directory on "
                           "command line")
    data_directory = sys.argv[1]
    # Call function to validate data in data directory
    validate_data(data_directory)


if __name__ == '__main__':
    # Python is running this file as a script, not importing it.
    main()

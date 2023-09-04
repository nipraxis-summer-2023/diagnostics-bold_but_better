""" Test script for SPM functions

Run these tests with::

    python3 findoutlie/tests/test_spm_funcs.py

or better, in IPython::

    %run findoutlie/tests/test_spm_funcs.py
"""

from pathlib import Path
import sys

MY_DIR = Path(__file__).parent
EXAMPLE_FILENAME = 'ds107_sub012_t1r2_small.nii'

CODE_DIR = (MY_DIR / '..').absolute()
sys.path.append(str(CODE_DIR))

import numpy as np

import nibabel as nib

# This import needs the directory containing the findoutlie directory
# on the Python path.
from spm_funcs import get_spm_globals, spm_global


def test_spm_globals():
    # Test get_spm_globals and spm_global functions
    example_path = MY_DIR / EXAMPLE_FILENAME
    expected_values = np.loadtxt(MY_DIR / 'global_signals.txt')
    glob_vals = get_spm_globals(example_path)
    assert glob_vals is not None, 'Did you forget to return the values?'
    assert np.allclose(glob_vals, expected_values, rtol=1e-4)
    img = nib.load(example_path)
    data = img.get_fdata()
    globals = []
    for vol_no in range(data.shape[-1]):
        vol = data[..., vol_no]
        globals.append(spm_global(vol))
    assert np.allclose(globals, expected_values, rtol=1e-4)


if __name__ == '__main__':
    # File being executed as a script
    test_spm_globals()
    print('Tests passed')

""" Test dvars implementation

You can run the tests from the root directory (containing ``README.md``) with::

    python3 -m pytest .
"""

import numpy as np

import nibabel as nib

import nipraxis as npx

from findoutlie.metrics import dvars


TEST_FNAME = npx.fetch_file('ds114_sub009_t2r1.nii')


def test_dvars():
    img = nib.load(TEST_FNAME)
    n_trs = img.shape[-1]
    n_voxels = np.prod(img.shape[:-1])
    dvals = dvars(img)
    assert len(dvals) == n_trs - 1
    # Calculate the values the long way round
    data = img.get_fdata()
    prev_vol = data[..., 0]
    long_dvals = []
    for i in range(1, n_trs):
        this_vol = data[..., i]
        d = this_vol - prev_vol
        long_dvals.append(np.sqrt(np.sum(d ** 2) / n_voxels))
        prev_vol = this_vol
    assert np.allclose(dvals, long_dvals)

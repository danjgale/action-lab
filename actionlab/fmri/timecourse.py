"""Handles run timecourse BOLD data.

 Module for performing BOLD manipulation (standardization, signal analysis,
filtering) and run splitting according to condtions/trial types.

"""

import numpy as np
import pandas as pd
import scipy
import nilearn

def percent_signal_change(x, baseline_index=None):
    """Compute percent signal change for each voxel (column) in array.
    Baseline must be specified and x must be a time * voxel array.
    """

    if baseline_index is not None:
        signal_mean = np.mean(x[baseline_index, :], axis=0)
    else:
        signal_mean = np.mean(x, axis=0)
    signal = 100 * ((x - signal_mean)/signal_mean)
    return signal


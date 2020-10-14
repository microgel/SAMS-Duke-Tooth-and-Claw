import numpy as np

def CORR_sort_and_delete_repeats(arr, return_indices=False):
    #sort arr and delete repeating elements
    if return_indices:
        indices =[idx for idx, item in enumerate(arr) if item in arr[:idx]]
        return np.unique(arr), indices
    else:
        return np.unique(arr)


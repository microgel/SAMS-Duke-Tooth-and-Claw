import numpy as np

def TEETH_remove_nans(A):
    temp = A[0]
    while(np.isnan(temp)):
        del A[0]
        temp = A[0]
    return A


import numpy as np
from scipy import sparse
import time

def EigenDecomp(H, numEigs):

    print("Renormalizing")
    sqrtInvD = sparse.csc_matrix(
        (
            1/np.sqrt(H.sum(1)),
            (np.arange(H.shape[1]),
            np.arange(H.shape[0]))
        )
    )
    H = sqrtInvD @ H @ sqrtInvD
    H = (H + H.conj().T)/2

    print("Computing eigenvalues")
    start_time = time.time()
    eig_lambda, U = sparse.linalg.eigsh(H, k=numEigs+1, maxiter=5000, return_eigenvectors=True)
    print("Eigen-decomp completed in %.2f seconds"%(time.time()-start_time))
    
    return U, eig_lambda


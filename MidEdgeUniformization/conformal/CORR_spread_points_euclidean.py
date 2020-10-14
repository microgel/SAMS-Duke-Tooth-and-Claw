import numpy as np
from tqdm.auto import tqdm

def CORR_spread_points_euclidean(X,seed_inds,n):
    """ calc n farthest points with seed_inds as seed set and Euclidean distance
    Parameters
    -----------
    X:  array_like
        arrays of vertices
    seed_inds:  array_like
                seed indices for somewhat randomization?
    n:  int
        farthest point indices?

    Returns
    -------
    sprd:   ndarray
            indices of distant points, shape(n,)
    """
    """ Safety check
    N = size(X,1);
    """
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    N = X.shape[0]

    """ Handle seed_inds if given
    dist = inf*ones(N,1);
    if(~isempty(seed_inds) )

        for k=1:length(seed_inds)
            tdist = sum((X - ones(N,1)*X(seed_inds(k),:)).^2,2);
            dist = min([dist tdist],[],2);
        end
        
    end
    """
    dist = np.inf*np.ones((N,))
    if(seed_inds):
        for k in range(seed_inds.shape[0]):
            tdist = np.sum((X - np.ones((N,1))*X[seed_inds[k]])**2, axis=1)
            dist = np.min(np.hstack([dist, tdist]), axis=1)
    
    """ Calculate spread indices. Used tqdm for progressbar
    sprd = zeros(n,1);
    progressbar
    for k=1:n
        progressbar(k/n);
        [~, tind] = max(dist);
        sprd(k) = tind(1);
        dist_from_k = sum((X - ones(N,1)*X(sprd(k),:)).^2,2);
        dist = min([dist dist_from_k],[],2);
    end
    """
    sprd = np.zeros((n,))
    for k in tqdm(range(n)):
        tind = np.argmax(dist, 0)
        sprd[k] = tind[0]
        dist_from_k =np.sum((X - np.ones((N,1))*X[sprd[k]])**2, axis=1)
        dist = np.min(np.hstack([dist, dist_from_k]),axis=1)
    
    return sprd


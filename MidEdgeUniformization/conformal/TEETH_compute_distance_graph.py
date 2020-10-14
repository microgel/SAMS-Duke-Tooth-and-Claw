import numpy as np
from scipy.sparse.csgraph import dijkstra

def TEETH_compute_distance_graph(A, tind):
    """calculate shortest path from point indices tind in a graph described by adj list A
    Parameters
    -----------
    A:  array_like
        Graph. shape(mxm)

    tind:   array_like
            Required source indices. shape(n,)
    Returns
    -------
    dists:  ndarray
            Distances of the nodes in tind to all other nodes.
            shape (n, m)
    """

    A = np.array(A) if not isinstance(A, np.ndarray) else A
    tind = np.array(tind) if not isinstance(tind, np.ndarray) else tind

    """
    n=size(A,2);
    m=length(tind);

    D = zeros(m,n);
    
    D is the return array, initialized here coz the matlab code 
    processed distances for each source separately
    """
    m = tind.shape[0]

    """Calculate all distances and then take needed indices
    for k=1:m
        [dists, path, pred]=graphshortestpath(A,tind(k), 'Directed', 'False' ,'Method', 'Dijkstra');
        D(k,:) = dists; 
    end
    """
    dists, preds, sources = dijkstra(A, directed=False)
    return dists.take(tind, axis=0)
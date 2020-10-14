import numpy as np
from .CORR_locate_midedge_boundary_vertices import CORR_locate_midedge_boundary_vertices

def CORR_find_midedge_boundary_connected_component(mF1, E2V_1):
    """find for each boundary *mid-edge* vertex a connected component
    Parameters
    -----------
    mF1:    array_like
            Triangle faces
    
    E2V_1:  array_like
            Edge or vertex coordinates
    Returns
    -------
    bcc:    ndarray
            indices with indices of connected components
    """

    """
    [mBoundary ring] = CORR_locate_midedge_boundary_vertices(mF1);
    """
    mBoundary, _ = CORR_locate_midedge_boundary_vertices(mF1)
    bound_len = mBoundary.shape[0]

    """each boundary mvertex is in its own connected component
    bcc = 1:length(mBoundary);
    """
    bcc = np.arange(bound_len)

    """all mid-edge boundary vertices tounch two vertices
    mev_2_v = E2V_1(mBoundary,:);
    """
    mev_2_v = E2V_1[mBoundary]

    """for every boundary me-v : make its component the same as all other who share a vertex with him
    for k=length(mBoundary):-1:1
        v = mev_2_v(k,:);
        tind = find(mev_2_v(:,1) == v(1) | mev_2_v(:,1) == v(2) | ...
            mev_2_v(:,2) == v(1) | mev_2_v(:,2) == v(2));
        aind=[];
        for j=1:length(tind)
            aind = [aind find(bcc == bcc(tind(j))) ];
        end
        bcc(aind) = k;
    end
    """
    for k in reversed(range(bound_len)):
        tind = np.where(np.any(mev_2_v==mev_2_v[k], axis=1))[0]
        aind = []
        for j in range(tind.shape[0]):
            aind+=np.where(bcc == bcc[tind[j]])[0].tolist()
        bcc[aind] = k
    return bcc

if __name__=="__main__":
    A=np.arange(6)
    A=np.array([A,A+3,A+5])
    E2V_1 = np.array([[-1, 0],
                    [2, 1],
                    [442, 1],
                    [432, 1432],
                    [42, 123],
                    [34, 33],
                    [4352, 3],
                    [12, 10],
                    [231, 10],
                    [-1, -100],
                    [-1, -100]])
    print(CORR_find_midedge_boundary_connected_component(A, E2V_1))
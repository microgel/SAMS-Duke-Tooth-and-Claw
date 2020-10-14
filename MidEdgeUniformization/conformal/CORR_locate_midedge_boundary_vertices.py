import numpy as np
from .CORR_compute_vertex_face_ring import CORR_compute_vertex_face_ring

def CORR_locate_midedge_boundary_vertices(mF):
    """
    Parameters
    -----------
    mF: array_like
        Triangle faces
    
    Returns
    -------
    mBoundary:  ndarray
                indices of boundary vertices?
    
    ring:   list
            faces adjacent to each vertex. 
            See CORR_compute_vertex_face_ring
    """

    mF = np.array(mF) if not isinstance(mF, np.ndarray) else mF
    """
    ring = CORR_compute_vertex_face_ring(mF);

    nver = max(max(mF));

    nmax = floor(nver/3);

    nmax not needed, it exists only to handle overflow and underflow in the matlab code...
    But proves to be useless in both cases
    """
    ring = CORR_compute_vertex_face_ring(mF)

    nver = np.max(mF)
    nver += 1

    """ Calculate all indices in ring with length<2
    mBoundary = zeros(1,nmax);
    end_ind=0;
    for k=1:nver
        if (length(ring{k}) < 2)
            end_ind = end_ind+1;
            mBoundary(end_ind) = k;
        end
    end

    mBoundary(end_ind+1:nmax) =[];
    """
    mBoundary = np.nonzero(np.array([len(ring[x]) for x in range(nver)]) < 2)[0]
    
    return mBoundary, ring

if __name__=="__main__":
    A=np.arange(6)
    A=np.array([A,A+1,A+2])
    print(CORR_locate_midedge_boundary_vertices(A))
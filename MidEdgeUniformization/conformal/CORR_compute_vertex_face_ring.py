import numpy as np
from .CORR_check_face_vertex import CORR_check_face_vertex

def CORR_compute_vertex_face_ring(face):
    """compute the faces adjacent to each vertex
    Parameters
    -----------
    face:   array_like
            Triangle faces
    
    Returns
    -------
    ring:   list
            faces adjacent to each vertex

    """

    """
    [tmp,face] = CORR_check_face_vertex([],face);

    nfaces = size(face,2);
    nverts = max(face(:));
    """
    _, face = CORR_check_face_vertex(np.empty(0), face)

    nfaces = face.shape[1]
    nverts = np.max(face)

    """ring is a list of variable sized lists.
    ring{nverts} = [];

    for i=1:nfaces
        for k=1:3
            ring{face(k,i)}(end+1) = i;
        end
    end
    """

    ring = [[] for i in range(nverts+1)]

    for i in range (nfaces):
        for k in range(3):
            ring[face[k,i]].append(i)

    return ring


if __name__=="__main__":
    A=np.arange(6)
    A=np.array([A,A+1,A+2])
    print(CORR_compute_vertex_face_ring(A))
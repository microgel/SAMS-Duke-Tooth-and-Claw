import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

def CORR_compute_laplacian_tension(vertex, face):
    """
    Parameters
    -----------
    vertex: array_like
            vertices
    face:   array_like
            faces
    
    Returns
    -------
    A:  scipy.sparse
        shape(nvertices, nvertices)

    """
    vertex = np.array(vertex) if not isinstance(vertex, np.ndarray) else vertex
    face = np.array(face) if not isinstance(face, np.ndarray) else face

    """
    nopts = size(vertex,1);
    notrg = size(face,1);
    """
    nopts = vertex.shape[0]
    notrg = face.shape[0]

    """ Code used for old approach, just kept it here if ever needed
    triang = face;
    coord = vertex;


    % tic;
    bir=[1 1 1];
    trgarea=zeros(1,notrg);
    for i=1:notrg
        trgx=[coord(triang(i,1),1) coord(triang(i,2),1) coord(triang(i,3),1)];
        trgy=[coord(triang(i,1),2) coord(triang(i,2),2) coord(triang(i,3),2)];
        trgz=[coord(triang(i,1),3) coord(triang(i,2),3) coord(triang(i,3),3)];
        aa=[trgx; trgy; bir];
        bb=[trgx; trgz; bir];
        cc=[trgy; trgz; bir];
        area=sqrt(det(aa)^2+det(bb)^2+det(cc)^2)/2;
        trgarea(i)=area;
    end


    %find the approximate voronoi area of each vertex
    AM = zeros(nopts, 1);
    for i=1:notrg
        AM(triang(i,1:3)) = AM(triang(i,1:3)) + trgarea(i)/3;
    end

    % T = sparse([1:nopts], [1:nopts], (AM), nopts, nopts, nopts);
    temp = 1./AM;
    temp = temp./max(temp);
    T = sparse([1:nopts], [1:nopts], temp, nopts, nopts, nopts);
    """
    # triang = face
    # coord = vertex

    # bir = np.ones((3,))
    # trgarea = np.zeros((notrg,))

    # for index, triangle in enumerate(coord[triang]):
    #     triangle = triangle.T
    #     trgarea[index] = np.sqrt(
    #         np.linalg.det(np.vstack([triangle[[0,1]], bir]))**2 +
    #         np.linalg.det(np.vstack([triangle[[0,2]], bir]))**2 +
    #         np.linalg.det(np.vstack([triangle[[1,2]], bir]))**2 
    #     )/2

    # AM = np.zeros((nopts,))
    # for i in range(notrg):
    #     AM[triang[i]]+=trgarea[i]/3

    # temp = 1/AM
    # temp = temp/np.max(temp)

    # T = csr_matrix((
    #     temp,
    #     (np.arange(nopts), np.arange(nopts))
    #     ),
    #     shape=(nopts, nopts)
    # )

    """ lil_matrix for faster element assignment
    A = sparse(nopts, nopts);
    """
    A = lil_matrix((nopts, nopts))

    """
    for i=1:notrg
        for ii=1:3
            for jj=(ii+1):3
                kk = 6 - ii - jj; % third vertex no
                v1 = triang(i,ii);
                v2 = triang(i,jj);
                v3 = triang(i,kk);
                e1 = [coord(v1,1) coord(v1,2) coord(v1,3)] - [coord(v2,1) coord(v2,2) coord(v2,3)];
                e2 = [coord(v2,1) coord(v2,2) coord(v2,3)] - [coord(v3,1) coord(v3,2) coord(v3,3)];
                e3 = [coord(v1,1) coord(v1,2) coord(v1,3)] - [coord(v3,1) coord(v3,2) coord(v3,3)];
                cosa = e2* e3'/sqrt(sum(e2.^2)*sum(e3.^2));
                sina = sqrt(1 - cosa^2);
                cota = cosa/sina;
                w = 0.5*cota;
                A(v1, v1) = A(v1, v1) - w;
                A(v1, v2) = A(v1, v2) + w;
                A(v2, v2) = A(v2, v2) - w;
                A(v2, v1) = A(v2, v1) + w;
            end
        end

    end
    L=A;
    return;
    """
    for i in range(notrg):
        for ii in range(3):
            for jj in range(ii+1, 3):
                kk = 3-ii-jj
                vs = face[i, [ii,jj,kk]]
                es = vertex[vs]-np.roll(vertex[vs],-1, axis=0)
                es[2] = np.negative(es[2])
                cosa = (es[1]@es[2].T)/np.sqrt(np.sum(es[1]**2)*np.sum(es[2]**2))
                sina = np.sqrt(1-cosa**2)
                cota = cosa/sina
                w = 0.5*cota
                A[vs[0],vs[0]]-=w
                A[vs[0],vs[1]]+=w
                A[vs[1],vs[1]]-=w
                A[vs[1],vs[0]]+=w

    return A


if __name__=="__main__":
    v = [[1, 2, 0], [1, 4, 0], [4, 2, 0], [3, 0, 1], [3, 1, 1], [0, 0, 1]]
    f = [[0, 1, 2], [3, 4, 5]]
    print(CORR_compute_laplacian_tension(v, f).todense())

import numpy as np

def ComputeMidEdge(G):
    """
    Parameters
    -----------
    G:  object, trimesh.base
        mesh object
    
    Returns
    -------
    mV: ndarray
        new vertices, shape(nume, 3)
    mF: ndarray
        new faces, shape same as old faces
    M:  ndarray
        Edge numbering for mapping edges back to edge index.
    e2v_map:    scipy.sparse matrix
                Edge to vertex map
                Edges in sparse bool COO graph format where connected vertices are True.
    """

    """ Since maltab returns (3,n) while trimesh gives (n,3). 
        All the ops in following code have direct indexing wrt the correct order of arrays
    V = G.V;
    F = G.F;
    """
    V = G.vertices
    F = G.faces

    """Most probably this, matlab code just returns a sparse matrix instead of the tree
    [M,e2v_map,~,~] = G.ComputeEdgeNumbering;
    """
    M = G.edges_sorted_tree.data
    e2v_map = G.edges_sparse

    """
    nume = size(G.V2E,2);
    numf = size(G.F,2);
    """
    nume = G.edges.shape[0]
    numf = G.faces.shape[0]

    """ the new vertices are the old edges
    mV = zeros(3,nume);
    for k = 1:nume
        vind = e2v_map(k,:);
        mV(:,k) = 0.5*(V(:,vind(1)) + V(:,vind(2)));
    end
    """
    mV = np.zeros((nume,3))
    for k in range(nume):
        vind = e2v_map[k]
        mV[k] = 0.5*(V[vind[0]]+V[vind[1]])
    
    """ the new faces are one per old face
    mF = zeros(3,numf);
    for k=1:numf
        f = F(:,k);
        
        ie1 = M(f(1),f(2));
        ie2 = M(f(2),f(3));
        ie3 = M(f(3),f(1));
            
        mF(:,k) = [ie1 ie2 ie3]';
    end

    None of the ops are explicitly transposed 
    since we conformed with trimesh format of V and F from the beginning
    The returned arrays also follow the same format
    """
    mF = np.zeros((numf,3))
    for k in range(numf):
        f = F[k]

        ie1 = M[f[0],f[1]]
        ie2 = M[f[1],f[2]]
        ie3 = M[f[2],f[0]]

        mF[k] = np.array([ie1, ie2, ie3])

    return mV, mF, M, e2v_map



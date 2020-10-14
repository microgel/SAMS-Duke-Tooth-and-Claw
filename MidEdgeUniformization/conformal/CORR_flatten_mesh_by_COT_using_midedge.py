import numpy as np
from .CORR_compute_laplacian_tension import CORR_compute_laplacian_tension
from .CORR_find_not_common import CORR_find_not_common

def CORR_flatten_mesh_by_COT_using_midedge(V,F,M,mV,mF,pmV, cut_face_ind):
    """ use cot-weights on the mesh to flatten it w.r.t. the flattened mid-edge mesh
    Parameters
    -----------
    V:  array_like
        vertices
    F:  array_like
        faces
    M:  array_like
        Edge numbering matrix
    mV: array_like
        midedge vertices
    mF: array_like
        midedge faces
    pmV:    ndarray
            the midpoint mesh and its embedding to plane
    cut_face_ind:   int
                    index of face where to cut the mesh
    
    Returns
    -------
    pV: ndarray
        Vertices of flattened mesh
    """
    #Converting array/matrix into numpy array
    # V = np.array(V)
    # F = np.array(F)
    # mV = np.array(mV)
    # mF = np.array(mF)
    # pmV = np.array(pmV)
    
    """
    [L] = CORR_compute_laplacian_tension(V,F);
    """
    L = CORR_compute_laplacian_tension(V,F)
    L = L.todense()
    
    """ for each vertex calculate its MVC in the space
        and use it to define its planar location
    pV = zeros(size(V,1),2);
    for k=1:size(V,1)
        v=V(k,:);
        Nind = find(M(k,:));%indices of the k-th vertex's neighbors
        MNind = M(k,Nind);
        %get the coordinate weight from the laplace matrix
        W=-L(k,Nind);
        W = W./L(k,k);    
        pV(k,:)=W*pmV(MNind,:);
    end
    """
    pV = np.zeros(V.shape[0],2)
    for k in range(0, V.shape[0]):
        v = V[k,:] 
        Nind = np.nonzero(M[k,:]) #indices of the k-th vertex's neighbors
        MNind = M[k, Nind]
                  
        #get the coordinate weight from the laplace matrix
        W = -L[k, Nind]
        W = W / L[k, k] 
        pV[k,:] = W * pmV[MNind,:]
                  
    """ for the vertices of the cut_face decide by mid-egde->vertices relation in that face
    mv = pmV(mF(cut_face_ind,:),:);
    A = 0.5*[1 0 1; 1 1 0; 0 1 1];
    B = [mv(2,:) ; mv(3,:); mv(1,:)];
    v = A\B;
    """
    mv = pmV[mF[cut_face_ind,:],:]
    A = 0.5*np.array(([1, 0, 1], [1, 1, 0], [0, 1, 1]))
    B = np.array([mv[1] , mv[2], mv[0]])
    v = np.linalg.lstsq(A,B)

    """ arrange the order
    mf = mF(cut_face_ind,:);
    f = F(cut_face_ind,:);
    for k=1:3
        if(k==3)
            k1=1;
            k2=2;
        elseif(k==2)
            k1=3;
            k2=1;
        else
            k1=2;
            k2=3;
        end
        ind1 = M(f(k),f(k1));
        ind2 = M(f(k),f(k2));
        find1 = find(mf == ind1);
        find2 = find(mf == ind2);
        m(k) = CORR_find_not_common([1 2 3], [find1 find2]);
    end
    """
    mf = mF[cut_face_ind]
    f = F[cut_face_ind]
    m=[]      
    for k in range(3):
        if(k==2):
            k1=0
            k2=1
        elif(k==1):
            k1=2
            k2=0
        else:
            k1=1
            k2=2
    
        ind1 = M[f[k],f[k1]]
        ind2 = M[f[k],f[k2]]
        find1 = np.nonzero(mf == ind1)
        find2 = np.nonzero(mf == ind2)
        m+=CORR_find_not_common([0,1,2], [find1, find2])

    """
    [sorted per] = sort(m);
    pV(F(cut_face_ind,per),:) = v;
    """
    per = np.argsort(m)
    pV[F[cut_face_ind,per]] = v

    return pV

    
    


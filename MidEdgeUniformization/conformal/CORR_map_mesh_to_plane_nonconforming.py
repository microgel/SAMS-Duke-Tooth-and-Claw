import numpy as np
from .CORR_compute_laplacian_tension import CORR_compute_laplacian_tension
from .CORR_calculate_conjugate_harmonic_faster import CORR_calculate_conjugate_harmonic_faster
from .CORR_get_midpoint_values_of_function import CORR_get_midpoint_values_of_function

def CORR_map_mesh_to_plane_nonconforming(V,F,mF,seed_face,M,E2V,numE, reflect_mesh=False):
    """ map the mesh V,F to plane using the non-conforming method of polthier 
        use the seed face to cut the mesh open
    
    Parameters
    -----------
    V:  array_like
        vertices
    F:  array_like
        faces
    mF: array_like
        midedge faces
    seed_face:  int
                index of face where to cut the mesh
    M:  array_like
        Edge numbering matrix
    E2V:    array_like
            Edges to vertices map
    numE:   int
            number of edges
    reflect_mesh:   Bool, optional
                    Return reflected mesh. Default False

    Returns
    -------
    pmV:    ndarray
            the midpoint mesh and its embedding to plane
    """
    #converting vector/matix into numpy array
    # V=np.array(V)
    # F=np.array(F)
    # mF=np.array(mF)
    # M=np.array(M)
    # E2V=np.array(E2V)
    # numE=np.array(numE)
    
    """ remember the face list for later
    oF = F;
    """
    oF = F
    
    """ cut out the seed face
    ioutf = seed_face;
    outf = F(ioutf,:);
    outf=sort(outf);
    F(ioutf,:)=[];
    """
    ioutf = seed_face
    outf = F[ioutf]
    outf.sort()
    F = np.delete(F, ioutf, axis=0)


    """
    [L] = CORR_compute_laplacian_tension(V,F);
    L1 = L;
    outf = sort(outf);
    L1(outf(1),:) = [];
    L1(outf(2)-1,:) = [];
    """
    L = CORR_compute_laplacian_tension(V,F)
    L = L.todense()
    L1 = L
    L1 = np.delete(L1, [outf[0], outf[1]-1], axis=0)
    
    """ Add 2 new rows
    L1rows = size(L1,1);

    L1(L1rows+1,outf(1)) = 1;
    L1(L1rows+2,outf(2)) = 1;
    """
    L1rows = L1.shape[0]

    L1 = np.vstack([L1, np.zeros((2, L1.shape[1]))])
    L1[L1rows,outf[0]] = 1
    L1[L1rows+1,outf[1]] = 1
    
    """
    b = zeros(L1rows,1);
    b(L1rows+1) = -1;
    b(L1rows+2) = 1;
    u = L1\b;
    """
    b = np.zeros((L1rows+2,))
    b[L1rows] = -1
    b[L1rows+1] = 1
    u = np.linalg.lstsq(L1,b)
    
    """ with linear system
        % [e_u_star] = CORR_calculate_conjugate_harmonic(F,V,u,M,E2V,numE);
        % withOUT linear system
    imissing_f = seed_face;
    [e_u_star] = CORR_calculate_conjugate_harmonic_faster(oF,V,mF,u,M,E2V,numE,imissing_f);
    """
    imissing_f = seed_face
    e_u_star = CORR_calculate_conjugate_harmonic_faster(oF,V,mF,u,M,E2V,numE,imissing_f)

    """
    [mu] = CORR_get_midpoint_values_of_function(oF,u, M, E2V, numE);
    """
    mu = CORR_get_midpoint_values_of_function(oF,u, M, E2V, numE)
    
    """
    if(reflect_mesh==0)
        pmV = [mu e_u_star]; 
    else
        pmV = [mu -e_u_star]; 
    end
    """
    pmV = np.array([mu, np.negative(e_u_star) if reflect_mesh else e_u_star])

    return pmV
    







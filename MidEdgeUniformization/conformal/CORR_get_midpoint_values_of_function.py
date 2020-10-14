import numpy as np

def CORR_get_midpoint_values_of_function(F, u, M, e2v, nume):
    """takes a function on a mesh and returns its values on the midedges (for the midedge mesh)
    mu = zeros(nume,1);

    for k=1:nume
    vind=e2v(k,:);
        mu(k) = 0.5*(u(vind(1),:) + u(vind(2),:));
    end 
    """

    mu = np.zeros((nume,))

    for k in range(nume+1):
        vind = e2v[k]
        mu[k] = np.sum(u[vind])/2

    return mu
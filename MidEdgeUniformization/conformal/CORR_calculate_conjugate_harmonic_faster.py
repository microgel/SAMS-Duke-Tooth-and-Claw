import numpy as np
from .CORR_compute_vertex_face_ring import CORR_compute_vertex_face_ring
from .CORR_set_local_conjugate_mv1_mv3 import CORR_set_local_conjugate_mv1_mv3

def CORR_calculate_conjugate_harmonic_faster(F, V, mF, u, M, e2v, nume, imissing_f):
    """ a procedure to calculate the conjugate harmonic function u_star to u on
        the mid-edges according to Polthier. This is to create conformal mapping
        of the mid edges.
        we use here the dual cot-formula

    Parameters
    -----------
    F:  array_like
        faces
    V:  array_like
        vertices
    mF: array_like
        midedge faces
    M:  array_like
        Edge numbering matrix
    e2v:    array_like
            Edges to vertices map
    nume:   int
            number of edges
    imissing_f: int
                index of face where to cut the mesh

    Returns
    -------
    e_u_star:   ndarray
                conformal mapping of the mid edges
    """

    """ construct the mid-edge vertex - mid-edge face ring (which has the same
        face indices as the original mesh
    ring = compute_vertex_face_ring(mF);
    """
    ring = CORR_compute_vertex_face_ring(mF)

    """ traverse the mesh starting for the first face
    tobe_visited = zeros(nume,1);
    tobe_visited(1) = 1;
    tobe_len = 1;
    e_u_star = NaN*ones(nume,1);
    e_u_star(M(F(tobe_visited(1),1),F(tobe_visited(1),2))) = 0;%set the addditive constant
    """
    tobe_visited = np.zeros((nume,))
    tobe_visited[0] = 1
    tobe_len = 0
    e_u_star = [None]*nume
    e_u_star[M[F[tobe_visited[0],0],F[tobe_visited[0],1]]] = 0

    """ while we haven't finished traversing the mesh
    """
    while(tobe_len>-1):
        """
        indf = tobe_visited(tobe_len);
        f = F(indf,:);
        tobe_len = tobe_len-1;
        """
        indf = tobe_visited[tobe_len]
        f = F[indf]
        tobe_len -= 1

        """ the three mid-edge vertices in this triangle\
        imv1 = M(f(2),f(3));
        imv2 = M(f(3),f(1));
        imv3 = M(f(1),f(2));
        """
        imvs = [
            M[f[1],f[2]],
            M[f[2],f[0]],
            M[f[0],f[1]]]
        
        """ add to the tobe_visited only the neighboring faces which in the common edge there is no data
        if (isnan(e_u_star(imv1))) %if the mid-edge vertex 1 has no data
            neigh_f=ring{imv1};
            if(neigh_f(1) == indf)
                neigh_f(1)=[];
            end
            if(~isempty(neigh_f))
                if (neigh_f(1) ~= imissing_f)%make sure we are not passing through the missing face
                    tobe_len = tobe_len+1;
                    tobe_visited(tobe_len) = neigh_f(1); %add this face to be visited
                end
            end
        end
        if (isnan(e_u_star(imv2))) %if the mid-edge vertex 2 has no data
            neigh_f=ring{imv2};
            if(neigh_f(1) == indf)
                neigh_f(1)=[];
            end
            if(~isempty(neigh_f))
                if (neigh_f(1) ~= imissing_f)
                    tobe_len = tobe_len+1;
                    tobe_visited(tobe_len) = neigh_f(1); %add this face to be visited
                end
            end
        end
        if (isnan(e_u_star(imv3))) %if the mid-edge vertex 3 has no data
            neigh_f=ring{imv3};
            if(neigh_f(1) == indf)
                neigh_f(1)=[];
            end
            if(~isempty(neigh_f))
                if (neigh_f(1) ~= imissing_f)
                    tobe_len = tobe_len+1;
                    tobe_visited(tobe_len) = neigh_f(1); %add this face to be visited
                end
            end
        end
        """
        for imv in imvs:
            if not e_u_star[imv]:
                neigh_f = ring[imv]
                if(neigh_f[0] == indf):
                    del neigh_f[0]
                if neigh_f and neigh_f[0] != imissing_f:
                    tobe_len += 1
                    tobe_visited[tobe_len] = neigh_f[0]
        """
        if (~isnan(e_u_star(imv1))) %if the mid-edge vertex 1 has data
            %dont change f
        elseif (~isnan(e_u_star(imv2)))
            f = [f(2) f(3) f(1)];
        elseif (~isnan(e_u_star(imv3)))
            f = [f(3) f(1) f(2)];
        else
            %not error - can happen
            %         disp('Error 2341 - stopping...')
            %         return
        end

        TODO: Can use np.roll(f, -1) and np.roll(f, 1) resp. 
        But confirm in testing if f is ALWAYS (3,)
        """
        
        if imvs[0]:
            pass 
        elif imvs[1]:
            f = np.array([f[1], f[2], f[0]])
        elif imvs[2]:
            f = np.array([f[2], f[0], f[1]])
        else:
            pass

        """ set the midedges vertices indices again
        imv1 = M(f(2),f(3));
        imv2 = M(f(3),f(1));
        imv3 = M(f(1),f(2));
        """
        imvs[0] = M[f[1], f[2]]
        imvs[1] = M[f[2], f[0]]
        imvs[2] = M[f[0], f[1]]

        """
        v1 = V(f(1),:); %the regular vertices
        v2 = V(f(2),:);
        v3 = V(f(3),:);
        u1 = u(f(1)); %the discrete harmonic values st the vertices v_i
        u2 = u(f(2)); %the discrete harmonic values
        u3 = u(f(3)); %the discrete harmonic values
        """
        vs = V[f[:3]]
        us = u[f[:3]]

        """ set the u_star at the other two vertices
        e_u_star(imv3) = CORR_set_local_conjugate_mv1_mv3(e_u_star(imv1),v1,v2,v3,u1,u2,u3);
        e_u_star(imv2) = CORR_set_local_conjugate_mv1_mv3(e_u_star(imv3),v3,v1,v2,u3,u1,u2);
        """
        e_u_star[imvs[2]] = CORR_set_local_conjugate_mv1_mv3(
            e_u_star(imvs[0]), 
            vs[0], vs[1], vs[2], 
            us[0], us[1], us[2])
        e_u_star[imvs[1]] = CORR_set_local_conjugate_mv1_mv3(
            e_u_star(imvs[2]), 
            vs[2], vs[0], vs[1], 
            us[2], us[0], us[1])

        return e_u_star #, M, e2v, nume





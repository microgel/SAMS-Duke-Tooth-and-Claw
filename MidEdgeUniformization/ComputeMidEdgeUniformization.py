import numpy as np
from scipy.spatial.distance import cdist
from ComputeMidEdge import ComputeMidEdge
from conformal.CORR_spread_points_euclidean import CORR_spread_points_euclidean
from conformal.CORR_map_mesh_to_plane_nonconforming import CORR_map_mesh_to_plane_nonconforming
from conformal.CORR_transform_to_disk_new_with_dijkstra import CORR_transform_to_disk_new_with_dijkstra
from conformal.CORR_flatten_mesh_by_COT_using_midedge import CORR_flatten_mesh_by_COT_using_midedge

def ComputeMidEdgeUniformization(G):
    """
    Parameters
    -----------
    G:  object, trimesh.base
        mesh object
        
    Returns
    --------
    uniV:   ndarray
            array of vertices shape
    uniF:   ndarray
            array of faces shape
    vertArea:   ndarray
                Areas formed by new vertices?
    """

    """ Surface areas of all triangles, transpose not needed
    [~,TriAreas] = G.ComputeSurfaceArea;
    G.Aux.VertArea = (TriAreas'*G.F2V)/3;
    vertArea = G.Aux.VertArea;
    """
    TriAreas = G.area_faces
    vertArea = (TriAreas@G.vertex_faces)/3

    """ compute mid-edge mesh
        Did not find similar func in trimesh obj, created one from matlab source code
    [mV,mF,M,E2Vmap] = G.ComputeMidEdge;
    """
    mV, mF, M, E2Vmap = ComputeMidEdge(G)
    E2Vmap = E2Vmap.todense()

    print('Find a face to cut the surface for uniformization...')
    """ decide where to cut the surface (for flattening/uniformization
        Not transposed coz maltab returns (3,n) while trimesh gives (n,3).
        Same reason applies for all code in this func.
    v_max_V = CORR_spread_points_euclidean(G.V',[],200);
    """
    v_max_V = CORR_spread_points_euclidean(G.vertices, [], 200)

    """ Pairwise distances
    GeoField = pdist2(real(G.V(:,v_max_V)'),real(G.V'));
    """
    GeoField = cdist((G.vertices[v_max_V]).real, G.vertices.real())

    """ Only argmin and the first nonzero index needed
    medianGeoField = mean(GeoField,2);
    [~, minplc] = min(medianGeoField);
    cut_vertex = v_max_V(minplc);
    cut_face = find(G.F2V(:,cut_vertex),1);
    """
    medianGeoField = np.mean(GeoField, 1)
    minplc = np.argmin(medianGeoField, axis=0)
    cut_vertex = v_max_V[minplc]
    cut_face = np.nonzero(G.vertex_faces[cut_vertex])[0]

    print('Flattening the mid-edge mesh...')
    """ flatten the mesh conformally
    unmV = CORR_map_mesh_to_plane_nonconforming(G.V',G.F',mF',cut_face,M,E2Vmap,G.nE,0);
    """
    unmV = CORR_map_mesh_to_plane_nonconforming(G.vertices, G.faces, mF, cut_face, M, E2Vmap, G.edges.shape[0])

    """ it is the same face number as the original mesh
    unmF = mF';
    unmF(cut_face,:) = [];
    """
    unmF = mF
    unmF = np.delete(unmF, cut_face, axis=0)

    """
    center_ind = cut_vertex;
    tind = find(G.F2V(:,center_ind),1);%v_max_V1(kk);
    center_ind = mF(1,tind);
    """
    center_ind = cut_vertex
    tind = np.nonzero(G.vertex_faces[center_ind])[0]
    center_ind = mF[tind,0]
    print('Flattened.')

    """ map domain to disk (add the infinity point back as sample point)
        transfer the indices of center point to the mid-edge
    unmV = CORR_transform_to_disk_new_with_dijkstra(unmV,mF,E2Vmap,center_ind);
    unmF = mF';
    """
    unmV = CORR_transform_to_disk_new_with_dijkstra(unmV, mF, E2Vmap, center_ind)
    unmF = mF
    
    print('Flattening the ORIGINAL mesh using the mid-edge flattening...')
    """ map the original mesh to the disk using the mid-edge structure
    uniV = CORR_flatten_mesh_by_COT_using_midedge(G.V',G.F',M,mV,unmF,unmV,cut_face)';
    uniV = [uniV;zeros(1,size(uniV,2))];
    uniF = G.F;
    """
    uniV = CORR_flatten_mesh_by_COT_using_midedge(G.vertices, G.faces, M, mV, unmF, unmV, cut_face)
    uniV = np.vstack([uniV, np.zeros((1, uniV.shape[1]))])
    uniF = G.faces

    return uniV, uniF, vertArea
import numpy as np
import numpy.matlib as matlib
from scipy.io import loadmat
import trimesh
import pickle
from scipy import sparse
import time
from tqdm import tqdm
from scipy.spatial.distance import cdist
import trimesh.triangles as triangles

def signT(p1,p2,p3):
    return (p1[0]-p3[0]) * (p2[1]-p3[1]) - (p2[0]-p3[0]) * (p1[1]-p3[1])

""" implementation from https://stackoverflow.com/a/2049593 """
def pointInTriangle(point, triangle):
    d1 = signT(point, triangle[0], triangle[1])
    d2 = signT(point, triangle[1], triangle[2])
    d3 = signT(point, triangle[2], triangle[0])

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not(has_neg and has_pos)

def MapSoftenKernel(Coords1, Coords2, F2, V1, V2, epsilon, augParam=1.5):
    nV1 = V1.shape[0]
    nV2 = V2.shape[0]

    # Will have to implement Matlab's pointLocation coz there's no equivalent in python
    # ti, BC = pointLocation(TR, Coords1)
    TR = trimesh.creation.extrude_triangulation(Coords2, F2)

    """ Could have sped up computation but need nan indexes for later processing """
    # Coords1 = Coords1[TR.contains(Coords1)]
    ti = []
    for coords in Coords1:
        for index, triangle in enumerate(TR.triangles):
            if pointInTriangle(coords, triangle):
                ti.append(index)
                break
        else:
            ti.append(np.nan)
    
    ti = np.array(ti)
    BC = triangles.points_to_barycentric(TR.triangles[ti[~np.isnan(ti)]], ti[~np.isnan(ti)])
    

    NaNInds = np.where(np.isnan(ti.flatten()))[0]
    BaseInds = np.arange(nV1)
    BaseInds = np.delete(BaseInds, NaNInds)

    # BC = np.delete(BC, NaNInds, axis=0) # not needed since we didnt calculate for nan values
    ti = np.delete(ti, NaNInds)

    BCTargets = TR.ConnectivityList[ti]

    BCPlan = sparse.csc_matrix(
        (
            BC.flatten(),
            (
                matlib.repmat(BaseInds.conj().T, 3,1),
                BCTargets.flatten()
            )
        ),
        shape=(nV1, nV2)
    )

    V1onV2 = V2 @ BCPlan.conj().T

    BCTargets_x = V2[BCTargets,0].reshape(BCTargets).conj().T
    BCTargets_y = V2[BCTargets,1].reshape(BCTargets).conj().T
    BCTargets_z = V2[BCTargets,2].reshape(BCTargets).conj().T

    Dists2BCTargets = np.sqrt(
                        (matlib.repmat(V1onV2[0],3,1)-BCTargets_x)**2 +
                        (matlib.repmat(V1onV2[1],3,1)-BCTargets_y)**2 +
                        (matlib.repmat(V1onV2[2],3,1)-BCTargets_z)**2
    )

    Threshold = np.zeros(nV1)
    Threshold[BaseInds] = augParam * Dists2BCTargets.mean(0)
    V1onV2 = V2 @ BCPlan.conj().T

    del BCPlan

    if epsilon == "auto":
        Edges = TR.edges
        epsilon = np.median(np.sum((V2[Edges[:,0]]-V2[Edges[:,1]])**2, axis=0), axis=0)/2
    
    Kernel_Dists2BCTargets = np.exp(-(Dists2BCTargets**2)/epsilon)
    Kernel12 = sparse.csc_matrix((
        Kernel_Dists2BCTargets.flatten(),
        (
            matlib.repmat(BaseInds.conj().T, 3,1),
            BCTargets.flatten()
        )),
        shape=(nV1, nV2)
    )

    Kernel_Dists2BCTargets = matlib.repmat(1/np.sum(Kernel_Dists2BCTargets, 0),3,1)* Kernel_Dists2BCTargets
    Transplan12 = sparse.csc_matrix(
        (
            Kernel_Dists2BCTargets.flatten(),
            (
                matlib.repmat(BaseInds.conj().T, 3,1),
                BCTargets.flatten()
            )
        ),
        shape=(nV1,nV2)
    )

    DistMatrix = cdist(V1onV2.conj().T, V2.conj().T)
    DistMatrix[DistMatrix>matlib.repmat(Threshold.conj().T, 1, DistMatrix.shape[0])] = np.nan

    AugKernel12 = sparse.csc_matrix(np.exp(-(DistMatrix**2)/epsilon))

    return Transplan12,Kernel12,AugKernel12,V1onV2


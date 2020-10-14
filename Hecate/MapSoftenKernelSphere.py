import numpy as np
import numpy.matlib as matlib
from scipy.io import loadmat
import trimesh
import pickle
from scipy import sparse
import time
from tqdm import tqdm
from scipy.spatial.distance import cdist

def MapSoftenKernelSphere(mapp, F2, V1, V2, epsilon, augParam=1.5):
    nV1 = V1.shape[0]
    nV2 = V2.shape[0]

    BaseInds = np.arange(nV1)

    V1onV2 = V2 @ mapp

    BCTargets = np.zeros((mapp.shape[0],3))

    for i in range(mapp.shape[0]):
        BCTargets[i] = np.nonzero(mapp[i])[0]
    


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
    V1onV2 = V2 @ mapp

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



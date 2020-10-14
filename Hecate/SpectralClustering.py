import numpy as np
import numpy.matlib as matlib
from scipy.io import loadmat
import trimesh
import pickle
from scipy import sparse
import time
from tqdm import tqdm

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Can probably reduce params by directly calling the funcs on meshList, if only needed one time for this func

# Depends upong how the final implementation is to be used
 
def SpectralClustering(meshList, vIdxArray, sqrtInvD, U, numSegments, kMeansMaxIter):
    np.nan_to_num(sqrtInvD, copy=False, nan=0)
    SignVectors = sqrtInvD@U[:, 1:]
    idx = KMeans(n_clusters=numSegments, max_iter=kMeansMaxIter).fit(SignVectors).labels_

    InfIdx = np.where(np.any(np.isnan(SignVectors), 1))[0]
    nVListCumsum = np.cumsum(diffMatrixSizeList)


    for j in range(len(InfIdx)):
        IdxJ = np.argmax(nVListCumsum>=InfIdx[j])
        ValidVList = np.arange(meshList[IdxJ].vertices.shape[0])
        IdxOnG = idx[vIdxArray[IdxJ,0]:vIdxArray[IdxJ,1]]
        ValidVList = np.delete(ValidVList, np.where(IdxOnG == idx[InfIdx[j]])[0])
        tmpDistMatrix = cdist(meshList[IdxJ].vertices[InfIdx[j]-vIdxArray[IdxJ,0]+1], 
                            meshList[IdxJ].vertices[ValidVList])
        minInd = np.argmin(tmpDistMatrix)
        idx[InfIdx[j]] = idx[ValidVList[minInd]+vIdxArray[IdxJ,0]-1]

    kIdx = idx
    return kIdx

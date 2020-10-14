import numpy as np
import numpy.matlib as matlib
from scipy.io import loadmat
import trimesh
import pickle
from scipy import sparse
import time
from tqdm import tqdm
from scipy.spatial.distance import cdist
import networkx as nx
from vertex_idx_cumsum import vertex_idx_cumsum


DATA_DIR="./"

# BNN = ?
# baseEps = ?
# fiberEpsVerts = ?

# Incase it doesn't exist
def ScaleArea(G):
    Center = np.mean(G.vertices, 1)
    G.vertices = G.vertices - matlib.repmat(Center, 1, G.vertices.shape[0])

    Area = G.area
    G.vertices = G.vertices*np.sqrt(1/Area)

    return G, Area, Center

def CreateDiffusionMatrixVertex():
    Names = None
    with open("Names.pkl", "rb") as names:
        Names = pickle.load(names)

    newMeshList = None
    with open(DATA_DIR+"newMeshList.pkl", "wb") as f:
        newMeshList = pickle.load(f)

    # dists = loadmat(os.path.join(DATA_DIR, "FinalDists.npy"))
    dists = np.load(os.path.join(DATA_DIR, "FinalDists.npy"))

    for i in range(len(newMeshList)):
        newMeshList[i], _, _ = ScaleArea(newMeshList[i])

    meshList = newMeshList

    triDists = np.triu(dists)
    baseEps = np.median(triDists[triDists>0])**2

    print("Find guess at fiber epsilon")

    edgeLengths = []
    for mesh in newMeshList:
        edgeLengts = np.concatenate([edgeLengths, mesh.edges_unique_length])

    fiberEpsVerts = np.median(edgeLengths)**2

    baseDistMatrix = dists

    nV = newMeshList[0].vertices.shape[0]
    H = sparse.csc_matrix(
        (
            len(newMeshList)*nV,
            len(newMeshList)*nV
        )
    )

    nnList = np.zeros((BNN, len(newMeshList)))
    for i in range(len(newMeshList)):
        idx = np.argsort(baseDistMatrix[:,i])
        nnList[:,i] = idx[1:(BNN+1)]

    print("Forming Diffusion")

    for i in range(len(newMeshList)):
        curAdj = cdist(newMeshList[i].vertices, newMeshList[i].vertices*nx.to_numpy_matrix(newMeshList[i].vertex_adjacency_graph))
        curAdj[curAdj==0] = np.nan
        curAdj = np.exp(-(curAdj**2)/fiberEpsVerts)
        curAdj = curAdj + np.identity(newMeshList[0].vertices.shape[0])
        curAdj = sparse.csc_matrix(
            np.diag(np.linalg.solve(np.sum(curAdj, axis=1), curAdj))
        )

        diffInds = np.nonzero(np.sum(nnList==i, axis=0))

        for index in diffInds:
            H[(b-1)*nV:b*nV, (i-1)*nV:i*nV] = np.exp(-(dists[i,b]**2)/baseEps)@curAdj

    for i in tqdm(range(len(newMeshList))):
        for j in range(i+1,len(newMeshList)):
            istart = i*nV
            iend = (i+1)*nV-1
            jstart = j*nV
            jend = (j+1)*nV-1

            H[istart:iend, jstart:jend] = np.maximum(
                H[istart:iend, jstart:jend].todense(), 
                H[jstart:jend, istart:iend].todense()
            )
            H[jstart:jend, istart:iend] = H[istart:iend, jstart:jend]

            if i%100 == 0:
                sparse.save_npz(DATA_DIR+"DiffusionMatrixVertex.npz", H)

    vIdxArray, vIdxCumSum = vertex_idx_cumsum(newMeshList)
    diffMatrixSize = vIdxCumSum[-1]
    diffMatrixSizeList = np.concatenate([[0],vIdxCumSum[:-1]])

    print("Saving Diffusion")
    sparse.save_npz(DATA_DIR+"DiffusionMatrixVertex.npz", H)
    return H, newMeshList, vIdxCumSum, vIdxArray
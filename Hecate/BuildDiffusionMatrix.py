import numpy as np
import numpy.matlib as matlib
from scipy.io import loadmat
import trimesh
import pickle
from scipy import sparse
import time
from tqdm import tqdm
from vertex_idx_cumsum import vertex_idx_cumsum

# if this works for our implementation then no need to put them as func params
# from constant import BNN, baseEps

def BuildDiffusionMatrix(BNN, baseEps, vIdxCumSum, dists=None):
    if not dists:
        dists = loadmat(os.path.join(DATA_DIR, "FinalDists.mat"))
        # or
        dists = np.load(os.path.join(DATA_DIR, "FinalDists.npy"))

    baseDistMatrix = dists - np.diag(np.diag(dists))
    n = dists.shape[0]
    rowNNs = baseDistMatrix.argsort(1)
    sDists = np.take_along_axis(baseDistMatrix, rowNNs, axis=1)

    sDists = sDists[:, 1:(BNN+1)]
    rowNNs = rowNNs[:, 1:(BNN+1)]

    baseWeights = sparse.csc_matrix(
            (
                sDists,
                (
                    matlib.repmat(np.arange(n).reshape(-1,1), 1, BNN),
                    rowNNs
                )
            ),
            shape=(n,n)
        )
    baseWeights = np.minimum(baseWeights, baseWeights.conj().T)
    for i in range(n):
        sDists[i] = baseWeights[i,rowNNs[i]]

    sDists = np.exp(-(sDists**2)/baseEps)

    #vIdxCumSum = vertex_idx_cumsum(meshList)

    diffMatrixSize = vIdxCumSum[-1]
    diffMatrixSizeList = np.concatenate([[0],vIdxCumSum[:-1]])
    diffMatrixRowIdx = []
    diffMatrixColIdx = []
    diffMatrixVal = []

    cBack = 0

    for j in tqdm(range(n)):
        softMapsMatrix = None
        # softMapsMatrix = np.load(os.path.join(DATA_DIR,"SoftMapsMatrix/SoftMapsMatrix_%d.npy"%j))
        with open(os.path.join(DATA_DIR,"SoftMapsMatrix/SoftMapsMatrix_%d.pkl"%j), "rb") as f:
            softMapsMatrix = pickle.load(f)
        for nns in range(BNN):
            if sDists[j,nns]==0:
                continue
            k = rowNNs[j,nns]

            AugKernel12 = softMapsMatrix[k]

            rowIdx, colIdx = np.nonzero(AugKernel12)
            val = AugKernel12[rowIdx, colIdx]

            diffMatrixRowIdx = np.concatenate([diffMatrixRowIdx, rowIdx+diffMatrixSizeList[j]])
            diffMatrixColIdx = np.concatenate([diffMatrixColIdx, colIdx+diffMatrixSizeList[k]])
            diffMatrixVal = np.concatenate([diffMatrixVal, sDists[j,nns]*val])

            rowIdx, colIdx = np.nonzero(AugKernel12.conj().T)
            val = AugKernel12[rowIdx, colIdx]

            diffMatrixRowIdx = np.concatenate([diffMatrixRowIdx, rowIdx+diffMatrixSizeList[j]])
            diffMatrixColIdx = np.concatenate([diffMatrixColIdx, colIdx+diffMatrixSizeList[k]])
            diffMatrixVal = np.concatenate([diffMatrixVal, sDists[j,nns]*val])

    H = sparse.csc_matrix(
        (
            diffMatrixVal,
            (
                diffMatrixRowIdx,
                diffMatrixColIdx
            )
        ),
        shape=(diffMatrixSize, diffMatrixSize)
    )

    sparse.save_npz(DATA_DIR+"DiffusionMatrix.npz", H)

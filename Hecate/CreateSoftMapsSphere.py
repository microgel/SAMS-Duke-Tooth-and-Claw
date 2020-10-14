import numpy as np
import numpy.matlib as matlib
from scipy.io import loadmat
import trimesh
import pickle
from scipy import sparse
import time
from tqdm import tqdm
from MapSoftenKernelSphere import MapSoftenKernelSphere

DATA_DIR="./"

def CreateSoftMapsSphere(fiberEps):
    Names = None
    with open("Names.pkl", "rb") as names:
        Names = pickle.load(names)

    newMeshList = None
    with open(DATA_DIR+"newMeshList.pkl", "wb") as f:
        newMeshList = pickle.load(f)

    HecateMaps = None
    with open(DATA_DIR+"HecateMaps.pkl", "wb") as f:
        HecateMaps = pickle.load(f)

    softMapsMatrix = []
    AugKernel12 = []
    AugKernel21 = []
    print("Creating soft maps matrix")

    for i in tqdm(range(len(meshList))):
        G1 = newMeshList[i]
        softMapsMatrixRow = []
        for j in range(len(meshList)):
            _, _ , temp, _ = MapSoftenKernelSphere(
                HecateMaps[j][i], 
                newMeshList[j].faces,  
                G1.vertices, 
                newMeshList[j].vertices, 
                fiberEps)
            AugKernel12.append(temp)

            _, _ , temp, _ = MapSoftenKernelSphere(
                HecateMaps[i][j],
                G1.faces, 
                newMeshList[j].vertices, 
                G1.vertices,
                fiberEps)
            AugKernel21.append(temp)

            softMapsMatrixRow.append(np.maximum(AugKernel12[j], AugKernel21[j].conj().T))
        
        softMapsMatrix.append(softMapsMatrixRow)
        AugKernel12.clear()
        AugKernel21.clear()

    with open(os.path.join(DATA_DIR+"softMapsMatrix.pkl"), "wb") as f:
        pickle.dump(softMapsMatrix, f)
        


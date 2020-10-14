import numpy as np
import numpy.matlib as matlib
from scipy.io import loadmat
import trimesh
import pickle
from scipy import sparse
import time
from tqdm import tqdm
from task.ComputeMidEdgeUniformization import ComputeMidEdgeUniformization
from MapSoftenKernel import MapSoftenKernel

DATA_DIR="./"

def CreateSoftMapsVertex(fiberEps):
    Names = None
    with open("Names.pkl", "rb") as names:
        Names = pickle.load(names)

    newMeshList = None
    with open(DATA_DIR+"newMeshList.pkl", "wb") as f:
        newMeshList = pickle.load(f)

    if not os.path.exists(DATA_DIR+"SoftMapsMatrixVertex/"):
        os.makedirs(DATA_DIR+"SoftMapsMatrixVertex/")


    """ Except for the weighted adjacency matrix part, its just a copy paste of CreateSoftMaps.m,
        They didn't even change the directory name.... And the weighted adjacency matrix isn't even used       anywhere
    """
    print("Creating soft maps matrix")

    for i in tqdm(range(len(meshList))):
        G1 = meshList[i]
        softMapsMatrix = []
        AugKernel12 = []
        AugKernel21 = []
        TextureCoordsSource = np.load(os.path.join(DATA_DIR, 'TextureCoordsSource/TextureCoordsSource_%d.npy'%i))
        
        for j in range(len(meshList)):
            _, _ , temp, _ = MapSoftenKernel(
                TextureCoordsSource[j], 
                ComputeMidEdgeUniformization(meshList[j])[0][0:2,:], 
                meshList[j].faces, 
                G1.vertices, 
                meshList[j].vertices, 
                fiberEps)
            AugKernel12.append(temp)

            _, _ , temp, _ = MapSoftenKernel(
                ComputeMidEdgeUniformization(meshList[j])[0][0:2,:],
                TextureCoordsSource[j], 
                G1.faces, 
                meshList[j].vertices, 
                G1.vertices,
                fiberEps)
            AugKernel21.append(temp)

            softMapsMatrix.append(np.maximum(AugKernel12[j], AugKernel21[j].conj().T))
            
        with open(os.path.join(DATA_DIR+"SoftMapsMatrixVertex/SoftMapsMatrixVertex_%d.pkl"%i), "wb") as f:
            pickle.dump(softMapsMatrix, f)
        


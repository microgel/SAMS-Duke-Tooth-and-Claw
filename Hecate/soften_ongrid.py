import os
import numpy as np
import numpy.matlib as matlib
from scipy.io import loadmat
import trimesh
import pickle
from scipy import sparse
import time
from tqdm import tqdm
from MapSoftenKernel import MapSoftenKernel

def soften_ongrid(G1, G2, soften_mat, TAXAind1, TAXAind2, options):
    cPSoftMapsMatrix = loadmat(soften_mat)

    # or
    cPSoftMapsMatrix = None
    with open(soften_mat, "rb") as soften_mat_f:
        cPSoftMapsMatrix = pickle.load(soften_mat_f)

    TextureCoords1Path = [options['TextureCoords1Path'], str(TAXAind1), "%d_%d.mat"%(TAXAind1, TAXAind2)]
    TextureCoords2Path = [options['TextureCoords2Path'], str(TAXAind1), "%d_%d.mat"%(TAXAind1, TAXAind2)]
    FibrEps = int(options['fibrEps'])

    G1 = trimesh.load(G1) if isinstance(G1, str) else G1
    G2 = trimesh.load(G2) if isinstance(G2, str) else G2

    TextureCoords1 = np.load(os.path.join(*TextureCoords1Path))
    TextureCoords2 = np.load(os.path.join(*TextureCoords2Path))

    _, _, AugKernel12, _ = MapSoftenKernel(TextureCoords1, TextureCoords2, G2.faces, G1.vertices, G2.vertices, FibrEps)
    _, _, AugKernel21, _ = MapSoftenKernel(TextureCoords2, TextureCoords1, G1.faces, G2.vertices, G1.vertices, FibrEps)

    temp = np.maximum(AugKernel12, AugKernel21.conj().T)
    cPSoftMapsMatrix[TAXAind1] = temp
    cPSoftMapsMatrix[TAXAind2] = temp
    with open(soften_mat, "wb") as soften_mat_f:
        pickle.dump(cPSoftMapsMatrix, soften_mat_f, protocol=pickle.HIGHEST_PROTOCOL)
    return cPSoftMapsMatrix

import os
import numpy as np
from CreateDiffusionMatrixVertex import CreateDiffusionMatrixVertex
from EigenDecomp import EigenDecomp
from SpectralClustering import SpectralClustering

# Most probably this file dedines all the required constants
from HecateSetup import BNN, numEigsVec, numSegmentsVec, kMeansMaxIter, dirCollate, colorSegments

# Most probably a class
from SegResult import SegResult

DATA_DIR="./"



if __name__ == "__main__":
    # HecateSetup()
    HECATE_DIR = os.path.join(DATA_DIR, "HecateVertex")
    if not os.path.exists(HECATE_DIR):
        os.makedirs(HECATE_DIR)

    
    H, newMeshList, vIdxCumSum, vIdxArray = CreateDiffusionMatrixVertex()
    cfg = {}
    cfg['dirCollate'] = dirCollate
    cfg['colorSegments'] = colorSegments

    for numEigs in numEigsVec:
        U, _, sqrtInvD = EigenDecomp(H, numEigs)
        for numSegments in numSegmentsVec:
            kIdx = SpectralClustering(newMeshList, vIdxArray, sqrtInvD, U, numSegments, kMeansMaxIter)
            out_path = [HECATE_DIR, "NumSegments_%d"%numSegments, "NumEigs_%d"%numEigs]
            cfg['out'] = os.path.join(*out_path)
            segRes = SegResult(newMeshList, kIdx, vIdxCumSum)
            segRes.calc_data()
            segRes.export(cfg)
            # Close all figures here
import trimesh
from datetime import datetime

class SAMSMesh(trimesh.Trimesh):
    def __init__(self, gaussianCurvature=None, RawArea=None, RawCenter=None, gaussMaxInds=None,
                meanCurvature=None, meanMinInds=None, DNE=None, DNEMaxInds=None, center=None, confFactorFaces=None,
                confFactorVertices=None, confMaxInds=None, *args, **kwargs):
        super(SAMSMesh, self).__init__(*args, **kwargs)
        self.gaussMaxInds = gaussMaxInds
        self.gaussianCurvature = gaussianCurvature
        self.meanCurvature = meanCurvature
        self.meanMinInds = meanMinInds
        self.DNE = DNE
        self.DNEMaxInds = DNEMaxInds
        self.center = center
        self.confFactorFaces = confFactorFaces
        self.confFactorVertices = confFactorVertices
        self.confMaxInds = confMaxInds

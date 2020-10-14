import trimesh
from SAMSMesh import SAMSMesh
from Setup.initialize import workingPath
import Setup.PreparationScripts as prep
from touch import touch

Names = prep.MoveDataToOutputFolder()

mesh = trimesh.load(workingPath + 'RawOFF/' + Names[0] + '.off')

sams = SAMSMesh(vertices=mesh.vertices, faces=mesh.faces, 
    face_normals=mesh.face_normals,vertex_normals=mesh.vertex_normals, 
    face_colors=mesh.face_attributes, vertex_colors=None, 
    face_attributes=None, vertex_attributes=mesh.vertex_attributes, 
    metadata=mesh.metadata, process=mesh.process, 
    validate=False, use_embree=True, initial_cache=None, visual=mesh.visual,
    gaussianCurvature=432, gaussMaxInds=2)


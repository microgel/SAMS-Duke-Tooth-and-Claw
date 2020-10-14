import numpy as np
def check_size(a):
    a = np.array(a)
    if a.size==0:
        return
    if (a.shape[0]>a.shape[1]):
        a = a.conj().transpose()
    
    if (a.shape[0]<3 and a.shape[1]==3):
        a = a.conj().transpose()
        
    if (a.shape[0]<=3 and a.shape[1]>=3 and np.sum(abs(a[:,2]))==0):
        # for flat triangles
        a = a.conj().transpose()
    if (a.shape[0]!=3 and a.shape[0]!=4):
        print('face or vertex is not of correct size')
        return
    return a
def CORR_check_face_vertex(vertex,face):

    #check_face_vertex - check that vertices and faces have the correct size
    #   [vertex,face] = check_face_vertex(vertex,face);
    #
    #   Copyright (c) 2007 Gabriel Peyre

    vertex = check_size(vertex)
    face = check_size(face)
    return vertex,face


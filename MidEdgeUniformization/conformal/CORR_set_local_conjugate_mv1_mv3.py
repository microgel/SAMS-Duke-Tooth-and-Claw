import numpy as np

def myangle(u,v):
    u = np.array(u)
    v = np.array(v)
    du = np.sqrt( np.sum(u**2) )
    dv = np.sqrt( np.sum(v**2) )
    #du = max(du,eps); dv = max(dv,eps);
    beta = np.arccos(np.sum(u*v)/(du*dv))
    return beta
    
def CORR_set_local_conjugate_mv1_mv3(u_star_mv1,v1,v2,v3,u1,u2,u3):
    u_star_mv1 = np.array(u_star_mv1)
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    alpha1 = myangle(v2-v1, v3-v1)
    #alpha2 = myangle(v3-v2, v1-v2)
    alpha3 = myangle(v1-v3, v2-v3)
    alpha1 = np.array(alpha1)
    u_star_mv3 = u_star_mv1 + 0.5*((u1-u2)/np.tan(alpha3) + (u3-u2)/np.tan(alpha1) )#the minus because 1->2 is inverse direction
    return u_star_mv3

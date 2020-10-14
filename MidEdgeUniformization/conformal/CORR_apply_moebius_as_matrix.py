import numpy as np
def CORR_apply_moebius_as_matrix(Mob,Z):

    #Z is Nx1 complex vector
    Mob = np.array(Mob)
    Z = np.array(Z)
    T_Z = (Mob[0,0]*Z+Mob[0,1])/(Mob[1,0]*Z+Mob[1,1])
    return T_Z




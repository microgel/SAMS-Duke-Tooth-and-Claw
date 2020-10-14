'''
Input:
v: nv x 3 vertex coordinates of a genus-0 triangle mesh
f: nf x 3 triangulations of a genus-0 triangle mesh
map: nv x 3 vertex coordinates of the mapping result

Output:
distortion: 3*nf x 1 angle differences

If you use this code in your own work, please cite the following paper:
[1] P. T. Choi, K. C. Lam, and L. M. Lui,
    "FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization for Genus-0 Closed Brain Surfaces."
    SIAM Journal on Imaging Sciences, vol. 8, no. 1, pp. 67-94, 2015.

Copyright (c) 2013-2018, Gary Pui-Tung Choi
https://scholar.harvard.edu/choi
'''

import numpy as np
import math
from scipy.sparse import csr_matrix

#assuming inputs of type ndarray object
def beltrami_coefficient(v, f, map):
    nf= len(f)

    #create required  matrix using np.array
    #reshaping that matrix using reshape
    Mi= np.array(range(1,nf+1), range(1,nf+1), range(1,nf+1)).reshape(1, 3*nf)
    
    #reshape transpose of f
    Mj= f.transpose().reshape(1, 3*nf)

    e1= v(f[:,3], range(1,3)) - v(f[:,2], range(1,3))
    e2= v(f[:,1], range(1,3)) - v(f[:,3], range(1,3))
    e3= v(f[:,2], range(1,3)) - v(f[:,1], range(1,3))

    area= np.multiply(-1, (np.multiply(e2(:,1) , e1(:,2)) + np.multiply(e1(:,1) , e2(:,2))))
    area= np.array(area,area,area).transpose()

    Mx= np.array(np.divide((e1(:,2),e2(:,2),e3(:,2)),area)/2).reshape(1, 3*nf)
    My= np.array(np.divide((e1(:,1),e2(:,1),e3(:,1)),area)/2).reshape(1, 3*nf)

    Dx= csr_matrix((Mi, (Mj, Mx)))
    Dy= csr_matrix((Mi, (Mj, My)))

    dXdu = Dx * map(:,1)
    dXdv = Dy * map(:,1)
    dYdu = Dx * map(:,2)
    dYdv = Dy * map(:,2)
    dZdu = Dx * map(:,3)
    dZdv = Dy * map(:,3)

    E = np.power(dXdu,2) + np.power(dYdu,2) + np.power(dZdu,2)
    G = np.power(dXdv,2) + np.power(dYdv,2) + np.power(dZdv,2)
    F = np.multiply(dXdu,dXdv) + np.multiply(dYdu,dYdv) + np.multiply(dZdu,dZdv)
    mu = np.divide((E - G + 2 * complex(0,1) * F),(E + G + 2*math.sqrt((np.multiply(E,G) - np.power(F,2)))))
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

def cotangent_laplacian(v, f):

nv = len(v);
f1 = f[:,0]
f2 = f[:,1]
f3 = f[:,2]




l1 = np.sqrt(np.sum(np.subtract(v[f2], v[f3])**2,1))
l2 = np.sqrt(np.sum(np.subtract(v[f3], v[f1])**2,1))
l3 = np.sqrt(np.sum(np.subtract(v[f1], v[f2])**2,1))

#using numpy advanced indexing to access all columns using the array f1, should work.


s = (l1 + l2 + l3)*0.5

area = np.sqrt( np.multiply(np.multiply(np.multiply(s,(s-l1)),(s-l2)),(s-l3)))



'''

nv = length(v);

f1 = f(:,1); f2 = f(:,2); f3 = f(:,3);

l1 = sqrt(sum((v(f2,:) - v(f3,:)).^2,2));
l2 = sqrt(sum((v(f3,:) - v(f1,:)).^2,2));
l3 = sqrt(sum((v(f1,:) - v(f2,:)).^2,2));

s = (l1 + l2 + l3)*0.5;
area = sqrt( s.*(s-l1).*(s-l2).*(s-l3));
 '''

cot12 = np.divide((l1**2 + l2**2 - l3**2),area/2);

cot23 = np.divide((l2**2 + l3**2 - l1**2),area/2);

cot31 = np.divide((l3**2 + l1**2 - l2**2),area/2);

diag1 = np.negative(cot12)-cot31
diag2 = np.negative(cot12)-cot23 
diag3 = np.negative(cot31)-cot23

II = [f1, f2, f2, f3, f3, f1, f1, f2, f3]
JJ = [f2, f1, f3, f2, f1, f3, f1, f2, f3]
V = [cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3]

'''
cot12 = (l1.^2 + l2.^2 - l3.^2)./area/2;
cot23 = (l2.^2 + l3.^2 - l1.^2)./area/2; 
cot31 = (l1.^2 + l3.^2 - l2.^2)./area/2; 
diag1 = -cot12-cot31; diag2 = -cot12-cot23; diag3 = -cot31-cot23;

'''
L = sparse(II,JJ,V,nv,nv);

return L
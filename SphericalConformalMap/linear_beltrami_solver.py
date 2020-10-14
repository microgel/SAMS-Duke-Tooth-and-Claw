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
from scipy.sparse import csr_matrix


def linear_beltrami_solver(v, f, mu, landmark, target):


    """ Convert to ndarray and Flatten expected vectors, to be safe """
    v = np.array(v)
    f = np.array(f)
    mu = np.array(mu).flatten()
    landmark = np.array(landmark).flatten()
    target = np.array(target)

    """ Uncomment if caller code did not take into account that python is zero indexed """
    # landmark -= 1
    # f -= 1

    """ Used for cleaner indexing """
    f = f.T

    """ Store the squared magnitude of mu, cleaner code.

    af = (1-2*real(mu)+abs(mu).^2)./(1.0-abs(mu).^2);
    bf = -2*imag(mu)./(1.0-abs(mu).^2);
    gf = (1+2*real(mu)+abs(mu).^2)./(1.0-abs(mu).^2);
    """
    mu_square = abs(mu)**2
    af = (1 - 2*mu.real + mu_square)/(1 - mu_square)
    bf = (-2*mu.imag)/(1 - mu_square)
    gf = (1 + 2*mu.real + mu_square)/(1 - mu_square)


    """ Triangle vertex coordinates, x and y for each. Get multiple sets from v.
    f0 = f(:,1); f1 = f(:,2); f2 = f(:,3);

    uxv0 = v(f1,2) - v(f2,2);
    uyv0 = v(f2,1) - v(f1,1);
    uxv1 = v(f2,2) - v(f0,2);
    uyv1 = v(f0,1) - v(f2,1); 
    uxv2 = v(f0,2) - v(f1,2);
    uyv2 = v(f1,1) - v(f0,1);
    """

    uxv0 = v[f[1],1] - v[f[2],1]
    uyv0 = v[f[2],0] - v[f[1],0]
    uxv1 = v[f[2],1] - v[f[0],1]
    uyv1 = v[f[0],0] - v[f[2],0]
    uxv2 = v[f[0],1] - v[f[1],1]
    uyv2 = v[f[1],0] - v[f[0],0]

    """ Calculate Area of triangle with Heron's formula. l is just an array of lengths of sides
    l = [sqrt(sum(uxv0.^2 + uyv0.^2,2)), sqrt(sum(uxv1.^2 + uyv1.^2,2)), sqrt(sum(uxv2.^2 + uyv2.^2,2))];
    s = sum(l,2)*0.5;
    area = sqrt(s.*(s-l(:,1)).*(s-l(:,2)).*(s-l(:,3)));

    Can probably use numpy polygon area calculation function to improve speed
    """ 
    l = [np.sqrt(uxv0**2 + uyv0**2), np.sqrt(uxv1**2 + uyv1**2), np.sqrt(uxv2**2 + uyv2**2)]
    s = np.sum(l, axis=0)/2
    area = np.sqrt(s*(s-l[0])*(s-l[1])*(s-l[2]))


    """ As is code from matlab """
    v00 = (af*uxv0*uxv0 + 2*bf*uxv0*uyv0 + gf*uyv0*uyv0)/area
    v11 = (af*uxv1*uxv1 + 2*bf*uxv1*uyv1 + gf*uyv1*uyv1)/area
    v22 = (af*uxv2*uxv2 + 2*bf*uxv2*uyv2 + gf*uyv2*uyv2)/area
    v01 = (af*uxv1*uxv0 + bf*uxv1*uyv0 + bf*uxv0*uyv1 + gf*uyv1*uyv0)/area
    v12 = (af*uxv2*uxv1 + bf*uxv2*uyv1 + bf*uxv1*uyv2 + gf*uyv2*uyv1)/area
    v20 = (af*uxv0*uxv2 + bf*uxv0*uyv2 + bf*uxv2*uyv0 + gf*uyv0*uyv2)/area

    """ Sparse matrix of... vertices?

    I = [f0;f1;f2;f0;f1;f1;f2;f2;f0];
    J = [f0;f1;f2;f1;f0;f2;f1;f0;f2];
    V = [v00;v11;v22;v01;v01;v12;v12;v20;v20]/2;
    A = sparse(I,J,-V);

    flattened coz scipy's csr doesn't implicitly flatten.
    """
    I = np.array([f[0],f[1],f[2],f[0],f[1],f[1],f[2],f[2],f[0]]).flatten()
    J = np.array([f[0],f[1],f[2],f[1],f[0],f[2],f[1],f[0],f[2]]).flatten()
    V = np.array([v00,v11,v22,v01,v01,v12,v12,v20,v20]).flatten()/2

    A = csr_matrix((-V, (I, J)))

    """ Fixed points to complex and some more math.

    targetc = target(:,1) + 1i*target(:,2);
    b = -A(:,landmark)*targetc;
    b(landmark) = targetc;
    A(landmark,:) = 0; A(:,landmark) = 0;
    A = A + sparse(landmark,landmark,ones(length(landmark),1), size(A,1), size(A,2));
    """
    
    targetc = target[:,0] + 1j*target[:,1]
    b = -A[:, landmark]*targetc
    b[landmark] = targetc
    A[landmark, :] = 0
    A[:, landmark] = 0
    A = A + csr_matrix((np.ones((landmark.shape[0], )), (landmark, landmark)), shape=A.shape)

    """ Solve Ax=B for x. used lstsq instead of solve.... more robust.
    map = A\b;
    map = [real(map),imag(map)];
    """
    spherical_map = np.linalg.solve(A.todense(),b,)
    spherical_map = np.array([spherical_map.real, spherical_map.imag])

    return spherical_map.T


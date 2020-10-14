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

import sys
import numpy as np
from scipy.sparse import csr_matrix, find


def spherical_tutte_map(f, bigtri=0):
    """
    Parameters
    -----------
    f:  array_like
        faces, or face indexes.
        shape (x,3)
    bigtri: int, optional
            index of the big triangle in the given faces. Value < 3. Default is 0(first)

    Returns
    -------
    st_map: ndarray
            Spherical Tutte Map(3x3) transposed
    """

    """ Safety checks """
    assert bigtri < 3, "bigtri must be less than 3"
    f = np.array(f) if not isinstance(f, np.ndarray) else f

    """ Add 1 to nv coz its used to create arange and matlab includes high value """
    nv = np.max(f) + 1
    nf = f.shape[0]

    """ Construct the Tutte Laplacian
    I = reshape(f',nf*3,1);
    J = reshape(f(:,[2 3 1])',nf*3,1);
    V = ones(nf*3,1)/2;
    W = sparse([I;J],[J;I],[V;V]);
    M = W + sparse(1:nv,1:nv,-diag(W)-(sum(W)'), nv, nv)

    f' in matlab is equivalent to f.conj().T
    Used order='F' in reshape instead of reversing order.
    """
    I = f.conj().T.reshape(nf * 3, 1, order='F')
    J = f[:, [1, 2, 0]].conj().T.reshape(nf * 3, 1, order='F')
    V = np.ones((nf * 3, 1)) / 2

    W = csr_matrix((
        np.array([V, V]).flatten(),
        (
            np.array([I, J]).flatten(),
            np.array([J, I]).flatten()
        )
    ))
    """ matrix.A1 just returns a flattened version """
    M = W + csr_matrix((
        -W.diagonal() - W.sum(0).conj().T.A1,
        (
            np.arange(nv),
            np.arange(nv)
        )),
        shape=(nv, nv)
    )

    """ 
    boundary = f(bigtri,1:3);
    [mrow,mcol,mval] = find(M(boundary,:));
    M = M - sparse(boundary(mrow), mcol, mval, nv, nv) + sparse(boundary, boundary, [1,1,1], nv, nv);
    """
    boundary = f[bigtri, :3]
    mrow, mcol, mval = find(M[boundary, :])
    M = M - csr_matrix((
        mval,
        (
            boundary[mrow],
            mcol
        )),
        shape=(nv, nv)
    ) + csr_matrix((
        [1, 1, 1],
        (
            boundary,
            boundary
        )),
        shape=(nv, nv)
    )

    """ boundary constraints for big triangle
    b = zeros(nv,1);
    b(boundary) = exp(1i*(2*pi*(0:2)/length(boundary)));
    """
    b = np.zeros(nv, dtype=np.complex)
    b[boundary] = np.exp(1j * (2 * np.pi * np.arange(3) / boundary.shape[0]))

    """ solve the Laplace equation to obtain a Tutte map. solve Ax=B for x
    z = M \ b;
    z = z-mean(z);
    """
    z = np.linalg.solve(M.todense(), b)
    z = z - np.mean(z)
    z_mags = np.abs(z) ** 2

    """ inverse stereographic projection
    S = [2*real(z)./(1+abs(z).^2), 2*imag(z)./(1+abs(z).^2), (-1+abs(z).^2)./(1+abs(z).^2)];
    """
    S = np.array([
        2 * z.real / (1 + z_mags),
        2 * z.imag / (1 + z_mags),
        (-1 + z_mags) / (1 + z_mags)
    ])

    """ Find optimal big triangle size
    w = complex(S(:,1)./(1+S(:,3)), S(:,2)./(1+S(:,3)));

    numpy broadcasting ftw
    """
    w = (S[0] + 1j * (S[1])) / (1 + S[2])

    """find the index of the southernmost triangle
    [~, index] = sort(abs(z(f(:,1)))+abs(z(f(:,2)))+abs(z(f(:,3))));
    inner = index(1);
    if inner == bigtri
        inner = index(2);
    end
    """
    index = np.argsort(np.abs(z[f]).sum(1))
    inner = index[1] if index[0] == bigtri else index[0]

    """Compute the size of the northern most and the southern most triangles 
    NorthTriSide = (abs(z(f(bigtri,1))-z(f(bigtri,2))) + ...
        abs(z(f(bigtri,2))-z(f(bigtri,3))) + ...
        abs(z(f(bigtri,3))-z(f(bigtri,1))))/3;

    SouthTriSide = (abs(w(f(inner,1))-w(f(inner,2))) + ...
        abs(w(f(inner,2))-w(f(inner,3))) + ...
        abs(w(f(inner,3))-w(f(inner,1))))/3;

    Shift right, subtract the arrays, take abs and sum
    """
    north_tri_side = np.sum(np.abs(z[f[bigtri]] - np.roll(z[f[bigtri]], 1))) / 3
    south_tri_side = np.sum(np.abs(w[f[inner]] - np.roll(w[f[inner]], 1))) / 3

    """rescale to get the best distribution
    z = z*(sqrt(NorthTriSide*SouthTriSide))/(NorthTriSide)
    """
    z = z * np.sqrt(north_tri_side * south_tri_side) / north_tri_side
    z_mags = np.abs(z) ** 2
    """inverse stereographic projection
    map = [2*real(z)./(1+abs(z).^2), 2*imag(z)./(1+abs(z).^2), (-1+abs(z).^2)./(1+abs(z).^2)];
    """
    st_map = np.array([
        2 * z.real / (1 + z_mags),
        2 * z.imag / (1 + z_mags),
        (-1 + z_mags) / (1 + z_mags)
    ])
    return st_map.T


if __name__ == "__main__":
    # f = np.array([[1,0,0],[0,1,2],[0,2,1], [0,2,2], [1,1,1]])
    f = [[0, 0, 0], [1, 0, 0], [0, 1, 2], [0, 2, 2], [1, 1, 1]]
    print(spherical_tutte_map(f, 1))


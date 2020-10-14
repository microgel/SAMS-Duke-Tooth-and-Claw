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
from operator import index
from unittest import result

import end as end
import numpy as np
import math

from .beltrami_coefficient import beltrami_coefficient
from .cotangent_laplacian import cotangent_laplacian
from .linear_beltrami_solver import linear_beltrami_solver
from .spherical_tutte_map import spherical_tutte_map, f


def spherical_conformal_map(v,f):

    #  temp = v(reshape(f',1,length(f)*3),1:3);

    f_transpose = np.reshape(np.transpose(f), (1, f.shape[0]*3)) 
    temp = []
    #assuming larger dimension of f is nf and not 3
    for element in f_transpose:
        temp.append(v[element])


    ''' 
    pseudocode: 
    nf rows of f are first rransposed to create a nf*3 matrix
    then this matrix is reshaped column wise to create a 1* (3*nf) array 
    the values in this row are used to access the rows of v [as an index], with all three columns
    this is temp.



    '''
    e1 = np.transpose(np.sqrt(np.sum(np.power((np.transpose(temp[1::3, 0:3] - temp[2::3, 0:3])),2), axis=0)))

    e2 = np.transpose(np.sqrt(np.sum(np.power((np.transpose(temp[0::3, 0:3] - temp[2::3, 1:3])),2), axis=0)))

    e3 = np.transpose(np.sqrt(np.sum(np.power((np.transpose(temp[0::3, 0:3] - temp[1::3, 1:3])),2), axis=0)))



    '''

    #e1 = sqrt(sum((temp(2:3:end,1:3) - temp(3:3:end,1:3))'.^2))';

    #e2 = sqrt(sum((temp(1:3:end,1:3) - temp(3:3:end,1:3))'.^2))';

    #e3 = sqrt(sum((temp(1:3:end,1:3) - temp(2:3:end,1:3))'.^2))';


    e1 =
    transpose of element-wise square root of each element in the row vector X (internal)

    X is the column wise sum of X2

    X2 is the result of element-wise squaring of transpose of the difference between two slices of the temp matrix

    the slices are defined as temp(1:3:end, 1:3) means start from 1 row, then skip 3 rows and so forth till the end and include all columns

    '''

    e_sum = np.add(np.add(e1, e2), e3)

    regularity = np.absolute(np.divide(e1, e_sum) - 1/3) + np.absolute(np.divide(e2, e_sum) - 1/3) + np.absolute(np.divide(e3, e_sum) - 1/3)

    bigtri = np.where(np.amin(regularity))
    # amin() in numpy returns the elemnt while min() matlab returns elemnt and index
    #where returns the index


    '''
    #regularity = abs(e1./(e1+e2+e3)-1/3)+...
    #    abs(e2./(e1+e2+e3)-1/3)+abs(e3./(e1+e2+e3)-1/3);
    #[~,bigtri] = min(regularity);

    regularity = 

    e1, e2, e3 are row vectors

    each of which is divided element-wise by the sum of all 3 and 1/3 is subtracted from it (an identity row vector multiplied by 1/3 is subtracted)

    the absolute of this operation is taken, and all three absolutes are added.

    the min() function returns two values: the lowest value of the array and its index
    the first value is disregarded and the second value is assigned to bigtri

    '''

    '''
    #North pole Step

    #nv = size(v,1); 
    # the length of v is stored in nv
    #M = cotangent_laplacian(v,f);

    ''' 

    nv = len(v)

    M = cotangent_laplacian(v, f)

    p1 = f[bigtri, 0]
    p2 = f[bigtri, 1]
    p3 = f[bigtri, 2]

    fixed = [p1, p2, p3]

    M_fixed = []

    for p in fixed:
        M_fixed.append(M[p])
    # because python doesn't allow slicing of matrices with an array.

    non_zero_indices = np.nonzero(M_fixed)
    #find() returns the non-zero elements in matlab, nonzero() in python returns the el
    [mrow, mcol, mval] = [M_fixed[non_zero_indices[0]], M_fixed[non_zero_indices[1]], M_fixed[non_zero_indices[2]]]

    #M = M - sparse(fixed[mrow],mcol,mval,nv,nv) + sparse(fixed,fixed,[1,1,1],nv,nv);

    #pretty weird on how to use matlab in python

    #trying to find numpy equivalent of find()

    #similarly finding and verifying sparse() in numpy and if its the same thing as in matlab


    '''

    p1 = f(bigtri,1);
    p2 = f(bigtri,2);
    p3 = f(bigtri,3);

    3 points taken

    fixed = [p1,p2,p3];

    [mrow,mcol,mval] = find(M(fixed,:));

    returns a 1x3 vector of non-zero elemnts in the fixed rows i guess?

    M = M - sparse(fixed(mrow),mcol,mval,nv,nv) + sparse(fixed,fixed,[1,1,1],nv,nv);

    sparse function from matlab with five arguments with an nv*nv sparse matrix produced

    ''' 

    x1 = 0 
    y1 = 0 
    x2 = 1 
    y2 = 0 # arbitrarily set the two points
    a = v[p2,0:3] - v[p1,0:3]
    b = v[p3,0:3] - v[p1,0:3]

    sin1 = (np.linalg.norm(np.cross(a,b), 2))/(np.linalg.norm(a,2)*np.linalg.norm(b,2))


    ori_h = np.linalg.norm(b,2)*sin1



    ratio = np.linalg.norm([x1-x2,y1-y2],2)/np.linalg.norm(a,2)
    y3 = ori_h*ratio

    x3 = math.sqrt(np.linalg.norm(b,2)^2*ratio^2-y3^2)


    # set the boundary condition for big triangle

    ''' 
    x1 = 0; 
    y1 = 0; 
    x2 = 1; 
    y2 = 0; % arbitrarily set the two points
    a = v(p2,1:3) - v(p1,1:3);
    b = v(p3,1:3) - v(p1,1:3);
    two slices of v based on p1, p2, p3

    sin1 = (norm(cross(a,b),2))/(norm(a,2)*norm(b,2));

    #norm(A) returns the largest singular value of A, max(svd(A)).
    cross product of a and b




    ori_h = norm(b,2)*sin1;



    ratio = norm([x1-x2,y1-y2],2)/norm(a,2);

    The largest singular value (same as norm(A)).



    y3 = ori_h*ratio; % compute the coordinates of the third vertex


    x3 = sqrt(norm(b,2)^2*ratio^2-y3^2);

    simple arithmetic
    '''
    c = np.zeros(nv,1)
    c[p1] = x1
    c[p2] = x2 
    c[p3] = x3
    d = np.zeros(nv,1) 
    d[p1] = y1
    d[p2] = y2
    d[p3] = y3
    z = np.linalg.lstsq(M, np.complex(c,d))
    #lstsq() vs \ in matlab can return different answers
    #unsure about this, please check.

    #creates a complex output from real inputs

    z = z - np.mean(z)




    '''

    % Solve the Laplace equation to obtain a harmonic map
    c = zeros(nv,1); 
    c(p1) = x1; 
    c(p2) = x2; 
    c(p3) = x3;
    d = zeros(nv,1); 
    d(p1) = y1; 
    d(p2) = y2; 
    d(p3) = y3;
    z = M \ complex(c,d);
    #creates a complex output from real inputs
    z = z-mean(z);



    ''' 

    #inverse stereographic projection
    S = [2*np.divide(np.real(z),(1+np.absolute(z)**2)), 2*np.divide(np.imag(z),(1+np.absolute(z)**2)), np.divide((-1+np.absolute(z)**2),(1+np.absolute(z)**2))]
    #./ may not exist in numpy, fix division
    w = np.complex( np.divide(S[:,0], np.add(1,S[:,2])), np.divide(S[:,1],(np.add(1,S[:,2]))))


    '''

    % inverse stereographic projection
    S = [2*real(z)./(1+abs(z).^2), 2*imag(z)./(1+abs(z).^2), (-1+abs(z).^2)./(1+abs(z).^2)];
    #some weird-ass arithmetic


    %% Find optimal big triangle size
    w = complex(S(:,1)./(1+S(:,3)), S(:,2)./(1+S(:,3)));
    '''


    inner = index[0]
    if inner == bigtri:
        inner = index[1]


    '''

    % find the index of the southernmost triangle
    [~, index] = sort(abs(z(f(:,1)))+abs(z(f(:,2)))+abs(z(f(:,3))));

    inner = index(1);
    if inner == bigtri
        inner = index(2);
    end
    ''' 
    NorthTriSide = (np.absolute(z[f[bigtri,0]]-z[f[bigtri,1]]) + np.absolute(z[f[bigtri,1]] - z[f[bigtri,2]]) + np.absolute(z[f[bigtri,2]]-z[f[bigtri,0]]))/3

    SouthTriSide = (np.absolute(w[f[inner,1]]-w[f[inner,2]]) + np.absolute(w[f[inner,2]]-w[f[inner,3]]) + np.absolute(w[f[inner,3]]-w[f[inner,1]]))/3



    ''' 
    % Compute the size of the northern most and the southern most triangles 
    NorthTriSide = (abs(z(f(bigtri,1))-z(f(bigtri,2))) + ...
        abs(z(f(bigtri,2))-z(f(bigtri,3))) + ...
        abs(z(f(bigtri,3))-z(f(bigtri,1))))/3;

    SouthTriSide = (abs(w(f(inner,1))-w(f(inner,2))) + ...
        abs(w(f(inner,2))-w(f(inner,3))) + ...
        abs(w(f(inner,3))-w(f(inner,1))))/3;

    '''
    z = z*(np.sqrt(NorthTriSide*SouthTriSide))/(NorthTriSide); 
    #is this an integer division/multiplication operation?
    # z is an array so make sure numpy divide works here lol

    '''
    z = z*(sqrt(NorthTriSide*SouthTriSide))/(NorthTriSide); 
    '''

    S = [np.divide(2*np.real(z),(1+np.absolute(z)**2)), np.divide(2*np.imag(z),(1+abs(z)**2)), np.divide((-1+abs(z)**2),(1+abs(z)**2))]


    '''
    % inverse stereographic projection
    S = [2*real(z)./(1+abs(z).^2), 2*imag(z)./(1+abs(z).^2), (-1+abs(z).^2)./(1+abs(z).^2)];

    #this is all directly translatably in python

    here we need spherical_tutte_map
    '''
    if (np.sum(np.isnan(S), axis=None) != 0):
        S = spherical_tutte_map(f,bigtri)
    #checks if any element in S is NaN and if so, runs stm()


    '''
    if sum(sum(isnan(S))) ~= 0
        % if harmonic map fails due to very bad triangulations, use tutte map
        S = spherical_tutte_map(f,bigtri); 
    end



    '''
    I = np.sort(S[:,3])
    #numpy only returns one sort() output while matlab returns two, the shape and the sorted array

    '''
    %% South pole step
    [~,I] = sort(S(:,3));

    '''

    def rounder(x):
        if (x-int(x) >= 0.5):
            return np.ceil(x)
        else:
            return np.floor(x)

    rounder_vec = np.vectorize(rounder)

    #rounding in numpy works differently than in MATLAB, where .5 is rounded up, while .5 can be both rounded down and up in numpy

    fixnum = max(rounder_vec(v.shape[0]/10),3)
    fixed = I[1:min(v.shape[0],fixnum)]
    #max and min are python functions, comparing the larger of two values, don't quite get why,
    # or what the point of using min there is


    ''''



    % number of points near the south pole to be fixed  
    % simply set it to be 1/10 of the total number of vertices (can be changed)
    % In case the spherical parameterization is not good, change 10 to
    % something smaller (e.g. 2)
    fixnum = max(round(length(v)/10),3);
    fixed = I(1:min(length(v),fixnum)); 






    % south pole stereographic projection
    P = [S(:,1)./(1+S(:,3)), S(:,2)./(1+S(:,3))]; 


    '''
    P = [(np.divide(S[:,0],(1+S[:,2]))), np.divide(S[:,1],(1+S[:,2]))] 


    mu = beltrami_coefficient(P, f, v)

    '''

    % compute the Beltrami coefficient
    mu = beltrami_coefficient(P, f, v); 

    '''


    spherical_map = linear_beltrami_solver(P,f,mu,fixed,P[fixed,:])
    #map is a keyword in python

    '''
    % compose the map with another quasi-conformal map to cancel the distortion
    map = linear_beltrami_solver(P,f,mu,fixed,P(fixed,:)); 


    '''


    if (np.sum(np.isnan(spherical_map), axis=None)) != 0:
        #% if the result has NaN entries, then most probably the number of
        #% boundary constraints is not large enough
        
        #% increase the number of boundary constrains and run again
        fixnum = fixnum*5 #% again, this number can be changed
        
        fixed = I[1:min(v.shape[0],fixnum)]
        spherical_map = linear_beltrami_solver(P,f,mu,fixed,P[fixed,:])
        
        if (np.sum(np.isnan(spherical_map), axis=None)) != 0:
            spherical_map = P #% use the old result
        


    '''

    if sum(sum(isnan(map))) ~= 0
        % if the result has NaN entries, then most probably the number of
        % boundary constraints is not large enough
        
        % increase the number of boundary constrains and run again
        fixnum = fixnum*5; % again, this number can be changed
        fixed = I(1:min(length(v),fixnum)); 
        map = linear_beltrami_solver(P,f,mu,fixed,P(fixed,:)); 
        
        if sum(sum(isnan(map))) ~= 0
            map = P; % use the old result
        end
    end


    '''

    z = np.complex(spherical_map[:,0], spherical_map[:,1])
    spherical_map = [2*np.divide(np.real(z),(1+np.absolute(z)**2)), 2*np.divide(np.imag(z),(1+np.absolute(z)**2)), - np.divide((np.absolute(z)**2-1),(1+np.absolute(z)**2))]
    #order of operations is a little confusing, although exponents come above division.

    return spherical_map
    '''

    z = complex(map(:,1),map(:,2));

    % inverse south pole stereographic projection
    map = [2*real(z)./(1+abs(z).^2), 2*imag(z)./(1+abs(z).^2), -(abs(z).^2-1)./(1+abs(z).^2)];
    end
    '''


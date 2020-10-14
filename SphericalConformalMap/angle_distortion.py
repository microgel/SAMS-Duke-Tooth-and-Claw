import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def expand_map(vertices):
    """ Convert to (nv,3) matrix

    if size(v,2) == 1
        v = [real(v),imag(v),zeros(length(v),1)];
    elseif size(v,2) == 2
        v = [v,zeros(length(v),1)];
    end
    """
    if len(vertices.shape) == 1:
        vertices=vertices.reshape(-1,1)

    if vertices.shape[1] == 1:
        return np.hstack([vertices.real, vertices.imag, np.zeros(shape=vertices.shape)])
    elif vertices.shape[1] == 2:
        return np.hstack([vertices, np.zeros(shape=(vertices.shape[0],1))])
    else:
        return vertices

def get_angles(v, f):
    """
    f1 = f(:,1); f2 = f(:,2); f3 = f(:,3);

    a3=[v(f1,1)-v(f3,1), v(f1,2)-v(f3,2), v(f1,3)-v(f3,3)];
    b3=[v(f2,1)-v(f3,1), v(f2,2)-v(f3,2), v(f2,3)-v(f3,3)];
    a1=[v(f2,1)-v(f1,1), v(f2,2)-v(f1,2), v(f2,3)-v(f1,3)];
    b1=[v(f3,1)-v(f1,1), v(f3,2)-v(f1,2), v(f3,3)-v(f1,3)];
    a2=[v(f3,1)-v(f2,1), v(f3,2)-v(f2,2), v(f3,3)-v(f2,3)];
    b2=[v(f1,1)-v(f2,1), v(f1,2)-v(f2,2), v(f1,3)-v(f2,3)];
    vcos1=(a1(:,1).*b1(:,1)+a1(:,2).*b1(:,2)+a1(:,3).*b1(:,3))./ ...
        (sqrt(a1(:,1).^2+a1(:,2).^2+a1(:,3).^2).*sqrt(b1(:,1).^2+b1(:,2).^2+b1(:,3).^2));
    vcos2=(a2(:,1).*b2(:,1)+a2(:,2).*b2(:,2)+a2(:,3).*b2(:,3))./...
        (sqrt(a2(:,1).^2+a2(:,2).^2+a2(:,3).^2).*sqrt(b2(:,1).^2+b2(:,2).^2+b2(:,3).^2));
    vcos3=(a3(:,1).*b3(:,1)+a3(:,2).*b3(:,2)+a3(:,3).*b3(:,3))./...
        (sqrt(a3(:,1).^2+a3(:,2).^2+a3(:,3).^2).*sqrt(b3(:,1).^2+b3(:,2).^2+b3(:,3).^2));

    Can use np.roll for calculating a,b
        a = np.roll(v[ft] - v[np.roll(ft, shift=1, axis=0)], shift=-1, axis=0)
    but it's faster only for very large nv and nf and also sacrifices clarity
    """
    ft = f.T
    a = np.array([v[ft[1]] - v[ft[0]], v[ft[2]] - v[ft[1]], v[ft[0]] - v[ft[2]]])

    """ Don't need to calculate b explicitly 
    b = np.array([v[ft[2]] - v[ft[0]], v[ft[0]] - v[ft[1]], v[ft[1]] - v[ft[2]]])
    """
    b = -np.roll(a, shift=1, axis=0)

    cos = np.sum(a*b, axis=2)/(np.linalg.norm(a, axis=2)*np.linalg.norm(b, axis=2))

    return cos

def angle_distortion(v, f, mapp, plot_hist=True):
    """ 
    Calculate and visualize the angle difference (angle_map - angle_v)

    Parameters
    -----------
    v:  array_like
        vertex coordinates of a genus-0 triangle mesh.
        shape (nv,3)
    f:  array_like
        triangulations of a genus-0 triangle mesh
        shape (nf,3)
    mapp:   array_like
        vertex coordinates of the mapping result
        shape(nv,3)
    plot_hist:  Boolean,
        Plot histogram. Default True.
    
    Returns
    -------
    distortion: ndarray
                angle differences.
                shape(3*nf,)
    """

    """ ndarray check """
    v = np.array(v) if not isinstance(v, np.ndarray) else v
    f = np.array(f) if not isinstance(f, np.ndarray) else f
    mapp = np.array(mapp) if not isinstance(mapp, np.ndarray) else mapp

    """
    nv = length(v);
    nv2 = length(map);

    if nv ~= nv2
        error('Error: The two meshes are of different size.');
    end
    """
    assert v.shape[0] == mapp.shape[0],f'Meshes are of different sizes with ${v.shape} and ${mapp.shape}'
    assert f.shape[1] >= 3, f'Error expect f to have shape(nf,3) instead got ${f.shape}'
    
    """ Get (nv,3) matrices from given input

    Repeating code, used a func.
    """
    v = expand_map(v)
    mapp = expand_map(mapp)
    
    """ Calculate angles

    Repeating code, used a func.
    """
    vcos = get_angles(v,f)
    mcos = get_angles(mapp, f)
    
    """ Calculate angle difference
    distortion = (acos([mapcos1;mapcos2;mapcos3])-acos([vcos1;vcos2;vcos3]))*180/pi;

    Flattened array expected
    """
    distortion = np.rad2deg(np.arccos(mcos)-np.arccos(vcos)).flatten()

    if(plot_hist):
        """ Uncomment and use just matplotlib if needed """
        # plt.hist(x=distortion, bins=np.arange(-180,181))
        sns.distplot(a=distortion, bins=np.arange(-180,181), kde=False)
        plt.xlim(-180, 180)
        plt.title('Angle Distortion')
        plt.xlabel('Angle difference (degree)')
        plt.ylabel('Number of angles')
        plt.show()

    return distortion


if __name__=="__main__":

    f = np.array([[0,1,2], [1,2,0], [2,0,1], [1,0,2]])
    v = np.array([1+0j,1j, 2+2j, 100+2j])
    # v = np.array([[1, 0, 2], [0, 1, 1], [2, 2, 2], [100, 3, 1]])
    mapp = np.array([[1,1],[0,1],[2,2], [100,1]])

    print(angle_distortion(v,f,mapp))

import numpy as np
import trimesh
from trimesh.visual import ColorVisuals, interpolate, create_visual


def plot_mesh(v, f, arg3=None, colormap='copper'):
    """
    Plot a mesh
    Parameters
    -----------
    v:  array_like
        vertex cooridnates.
        shape (nv,3)
    f:  array_like
        Faces, triangulations.
        shape (nf,3)
    arg3:  array_like, optional
        Color intensities for vertices. Default with show [0.6,1,1] as facecolor for all faces
        shape (nv,)
    colormap: string, optional
        colormap to use. Only useful if arg3 intensities are specified. Default 'copper'
    """

    """ Define vertex colors if intensities given else default to constant facecolor """
    vertex_colors = interpolate(arg3, color_map=colormap) if arg3 is not None else None
    face_colors = [0.6,1,1] if arg3 is None else None

    mesh = trimesh.Trimesh(vertices = v,
                        faces = f,
                        vertex_colors = vertex_colors,
                        face_colors = face_colors,
                        )

    mesh.show()

if __name__=="__main__":
    v = [[1, 2, 0], [1, 4, 0], [4, 2, 0], [3, 0, 1], [3, 1, 1], [0, 0, 1]]
    f = [[0, 1, 2], [3, 4, 5]]
    plot_mesh(v,f, [0,1,2,3,4,5])
import os
import sys
import csv

import numpy as np
import numpy.matlib
import trimesh
import networkx as nx
import copy
import warnings
import pickle

from operator import itemgetter
from scipy import sparse, spatial
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from itertools import chain 

from .MappingSetup import *
from .initialize import workingPath
from SAMSMesh import SAMSMesh
# from SphericalConformalMap.spherical_conformal_map import spherical_conformal_map

from touch import touch

from datetime import datetime



def getNames(path=workingPath):
    Names = []
    if os.path.exists(path + 'Names.csv'):
        with open(path + 'Names.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                Names = row
    else:
        raise Exception('Names file does not exist, run MoveDataToOutputFolder first!')
    if not Names:
        raise Exception('No names found, run MoveDataToOutputFolder first or\
                        add meshes to data path!')

    return Names

def MoveDataToOutputFolder(inputPath=dataPath, outputPath=workingPath):
    """
    Moves the input data to ouput working path

    Parameters
    ------------
    None, but note data path must have mesh files with extensions
    binary/ASCII STL, Wavefront OBJ, ASCII OFF, binary/ASCII PLY, 
    GLTF/GLB 2.0, 3MF, XAML, 3DXML, etc.

    Returns
    ----------
    Names: list string
      Names of the meshes in the data path
    """

    # function starts filling in new working directory and extracts basic info
    rawOFFPath = outputPath + 'RawOFF\\'
    rawMATPath = outputPath + 'RawPLY\\'
    touch(rawOFFPath)
    touch(rawMATPath)

    names = []
    for file in os.listdir(inputPath):
        if file.endswith(".off") or file.endswith(".obj") or file.endswith(".ply"):
            filename = os.path.splitext(file)[0]
            mesh = trimesh.load(inputPath + file)
            mesh.export(rawOFFPath + filename + '.off')
            mesh.export(rawMATPath + filename + '.stl')
            names.append(filename)

    with open(outputPath + 'Names.csv', 'w', newline='') as csvfile:
        namewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        namewriter.writerow(names)

    return names


def ComputeF2V(mesh):
    nv = len(mesh.vertices)
    nf = len(mesh.faces)
    
    # F2V_list = np.zeros((nf, nv))
    # for j, faces in enumerate(mesh.vertex_faces):
    #     for face in faces:
    #         if face != -1:
    #             F2V_list[face][j] = 1
    # F2V = sparse.csc_matrix(F2V_list)
    
    I = np.reshape(mesh.faces,(3*nf,1)).flatten()
    J = np.array([list(range(nf))*3]).flatten()
    J.sort()
    F2V = sparse.coo_matrix((np.ones(3*nf),(J,I)),shape=(nf,nv))
    
    return F2V

def GetGPLmk(mesh, numLmk, gaussian_curvature, mean_curvature, 
            triarea, lam = 0.5, rho = 1, epsilon = 0.2):
    
    gaussian_curvature= np.array(gaussian_curvature,ndmin=2).T
    mean_curvature = np.array(mean_curvature,ndmin=2).T
    num_vertices = len(mesh.vertices)
    startTime = datetime.now()
    Faces_to_vertex = np.transpose(ComputeF2V(mesh))
    
    triarea = np.reshape(triarea, (len(triarea), 1))
    voronoi_areas = Faces_to_vertex.dot(triarea)
    #Lambda = np.multiply(voronoi_areas, (lam * np.power(np.abs(gaussian_curvature),rho) / np.power(np.sum(np.abs(gaussian_curvature)),rho)) 
    #         + ((1 - lam) * np.power(np.abs(mean_curvature),rho) / np.power(np.sum(np.abs(mean_curvature)),rho)))
    Lambda = np.multiply(voronoi_areas,4*np.power(mean_curvature,2)-2*gaussian_curvature)
    bandwidth = np.mean(mesh.edges_unique_length) / 5

    BNN = min(100, num_vertices)
    
    nbrs = NearestNeighbors(n_neighbors=BNN + 1, algorithm='ball_tree').fit(mesh.vertices)
    distances, indices = nbrs.kneighbors(mesh.vertices)

    distances = distances.reshape(1, num_vertices * (BNN + 1)).squeeze()
    indices = indices.reshape(1, num_vertices * (BNN + 1)).squeeze()
    rows = np.matlib.repmat(range(num_vertices), 1, BNN+1).squeeze()
    fullPhi = sparse.coo_matrix((np.exp(-np.power(distances, 2) / bandwidth), 
                                (rows, indices)), shape=(num_vertices, num_vertices))
    fullPhi = fullPhi.tolil()

    fullMatProd = fullPhi.T@(sparse.spdiags(Lambda.T,0,num_vertices,num_vertices))@fullPhi

    print('fullMatProd Done')

    KernelTrace = fullMatProd.diagonal()
    GPLmkIdx = []

    invKn = np.zeros((numLmk, numLmk))


    cback = 0
    for i in range(numLmk):
        print(i,flush=True)
        if i == 0:
            ptuq = KernelTrace
        elif i == 1:
            #fullMatProd = fullMatProd.tolil()
            invKn[0:i, 0:i] = 1 / fullMatProd[GPLmkIdx[0], GPLmkIdx[0]]
            ptuq = KernelTrace - np.sum(np.multiply(np.transpose(fullMatProd[:, GPLmkIdx].todense()), 
                (invKn[0:i, 0:i]*(fullMatProd[GPLmkIdx,:].todense()))),axis=0).flatten()
        else:
            p = fullMatProd[GPLmkIdx[0:i-1], GPLmkIdx[i-1]].todense()
            conjugate = (np.transpose(p).dot(invKn[0:i-1, 0:i-1])).dot(p)
            mu = 1 / (fullMatProd[GPLmkIdx[i-1], GPLmkIdx[i-1]] - conjugate)
            myMatrix = np.concatenate((np.eye(i-1) + mu.item()* (p.dot(np.transpose(p))) * invKn[0:i-1, 0:i-1], -(mu.item())*p),axis=1)
            invKn[0:i-1, 0:i] = invKn[0:i-1, 0:i-1].dot(myMatrix)
            invKn[i-1, 0:i] = np.concatenate((np.transpose(np.array(invKn[0:i-1, i],ndmin=2)), np.array(mu))).flatten()
            productEntity = invKn[0:i, 0:i].dot(fullMatProd[GPLmkIdx, :].todense())
            ptuq = KernelTrace - np.sum(np.multiply(np.transpose(fullMatProd[:, GPLmkIdx].todense()),
                productEntity),axis=0).flatten()
        
        curIdx = -1
        if len(ptuq) == 1:
            ptuq = ptuq.tolist()[0]
        sortIdx = np.argsort(ptuq)
        #THIS IS A DUMB FIX BECAUSE NUMPY IS STUPID
        #SOM
        while True:
            if sortIdx[curIdx] in GPLmkIdx:
                curIdx = curIdx - 1
            else:
                GPLmkIdx.append(sortIdx[curIdx])
                break
        
        if i == numLmk-1:
            return GPLmkIdx
            
def FindOrientedBoundaries(mesh):
    """
    Computes the list of boundaries in order

    Parameters
    ------------
    mesh: trimesh mesh
      The mesh to find boundaries

    Returns
    ----------
    BL: list (number boundaries, n)
      The boundaries in order
    BI: list (1, n)
      The next boundary in the chain
    """
    BL = []
    BI = []
    if mesh.is_watertight:
        return BL, BI
    
    num_vertices = len(mesh.vertices)
    edge_adjacency = mesh.edges_sparse.tolil()

    edge_i, edge_j = edge_adjacency.nonzero()

    BV = sparse.lil_matrix((num_vertices,1))
    BE = sparse.lil_matrix((num_vertices,2))
    for m in range(len(edge_i)):
        i = edge_i[m]
        j = edge_j[m]
        if bool(edge_adjacency[i, j]) != bool(edge_adjacency[j, i]):   # xor i and j
            boolean_matrix = (BE[i,:] == 0)
            loc = np.where(boolean_matrix.toarray())[1]
            if loc.size != 0:
                BE[i, loc[0]] = j
            boolean_matrix = (BE[j,:] == 0)
            loc = np.where(boolean_matrix.toarray())[1]
            if loc.size != 0:
                BE[j, loc[0]] = i
            BV[i] = 1
            BV[j] = 1
    BV = BV.nonzero()[0]
    
    bn = 0
    while np.count_nonzero(BV) > 0:
        vind = next((i for i, x in enumerate(BV) if x), None)   # Find first nonzero element
        v = BV[vind]
        BE[v, 1] = 0
        i = 0
        hole_boundaries = []
        
        while v != 0:
            hole_boundaries.append(int(v))
            i += 1
            vind = np.where(BV == v)
            for j in vind:
                BV[j] = 0
            BE[BE == v] = 0
            nonzero_BE = BE[v, :].nonzero()[1]
            if nonzero_BE.size == 0:
                break
            v = BE[v, nonzero_BE][0, 0]
            
        BL.append(hole_boundaries)
        bn += 1

    for i in range(bn): # i=1:bn-1
        BLi = BL[i][:]
        ind= np.nonzero(BLi)
        hole_boundaries_next = np.zeros(np.size(mesh.vertices))
        for position, index in enumerate(BL[i]):
            hole_boundaries_next[index] = position
        BI.append(hole_boundaries_next)
        
    return BL, BI




def getPoints(mesh, indices, radius):    
    vertices = {}
    num_vertices = len(mesh.vertices)

    
    if radius == 0:
        for ind in indices:
            vertices[ind] = []
        return vertices

    else:
        adjacency_matrix = nx.adjacency_matrix(mesh.vertex_adjacency_graph, 
                                            nodelist=range(num_vertices))
        power = (adjacency_matrix ** radius) + (adjacency_matrix ** (radius-1))   # Depth-adjacency
        AK = power + sparse.eye(num_vertices, num_vertices)   # Depth-adjacency + self
        AK[AK>0]=1
        for ind in indices:
            vertices[ind] = np.nonzero(AK[ind, :])[1].tolist()
    
    return vertices


def visualizePointsOnMesh(mesh, ptCloud = None, radius = 0):
    '''
    Visualizes a mesh, with optional point cloud to highlight
    mesh vertices

    Parameters
    ------------
    mesh: trimesh mesh
      The mesh to be visualized
    ptCloud: (n) list of ints
      Mesh vertex indices that are to be highlighted
    radius: int
      Discrete radius of how many points away to look for
    '''
    if ptCloud is not None:
        neighbors = getPoints(mesh, ptCloud, radius)
        ptCloudFcn = [0] * len(mesh.vertices)
        for vertex in neighbors.keys():
            ptCloudFcn[vertex] = 1
            for ind in neighbors[vertex]:
                ptCloudFcn[ind] = 1
                
        mesh.visual.vertex_colors = trimesh.visual.interpolate(ptCloudFcn, color_map='jet')

    scene = trimesh.Scene(mesh)
    for light in scene.lights:
        light.intensity = sceneBrightness
    scene.show()


def isManifold(mesh, boundary):
    '''
    ISMANIFOLD returns true if the mesh is a manifold, false otherwise.

    Parameters
    ------------
    mesh: trimesh mesh
      The mesh to check
    boundary: (number boundaries, n)
      Index of mesh vertices that correspond to the boundary

    Returns
    ----------
    meshData: dictionary
      Results isManifold
        manifold: boolean
          True if the mesh is a manifold, false otherwise
        nonManifoldEdges: list (number of edges, n)
          Indices of all the edges with more than 
          two adjacent faces
        nonManifoldVertices: list
          All the vertices such that their 1-ring is 
          not a disk
        nonManifoldBoundary: list
          All the boundary vertices such that their 
          1-ring is not a disk (aka they have more than
          two neighbours on the boundary)
          
    
    Created in MATLAB by Nave Zelzer on march 31 2014
    Ported to Python by Rob Yale on July 13, 2020
    '''
    numFaces = len(mesh.faces)
    numVertices = len(mesh.vertices)
    Faces = mesh.faces.transpose()
    face2Vertex = ComputeF2V(mesh).tolil()
    meshData = {}
    meshData['manifold'] = True
    meshData['nonManifoldEdges'] = []
    meshData['nonManifoldVertices'] = []
    meshData['nonManifoldBoundary'] = []
    


    # we have three checks to do:
    # 1 ) check that every edge only have one or two faces adjacent to it, no
    #     more nor less!!!
    # 
    # 2 ) check that the mesh don't have an hourglass shape a.k.a exist a
    #     vertex with 2-disk neighbourhood.
    # 
    # 3 ) check that the every boundary vertex have only two neighbours on the
    #     boundary. aka the topology of the boundary is non disk shape.


    # step 1:
    # 1 ) check that every edge only have one or two faces adjacent to it, no
    #     more nor less!!!
    I = []
    J = []
    I.extend(Faces[0,:]); I.extend(Faces[1,:]); I.extend(Faces[2,:])
    I.extend(Faces[1,:]); I.extend(Faces[2,:]); I.extend(Faces[0,:])
    J.extend(Faces[1,:]); J.extend(Faces[2,:]); J.extend(Faces[0,:])
    J.extend(Faces[0,:]); J.extend(Faces[1,:]); J.extend(Faces[2,:])
    S = np.ones(6 * numFaces)
    edges = sparse.coo_matrix((S, (I,J)))
    [i,j] = np.where((edges > 2).toarray())
    if len(i) > 1:
        meshData['manifold'] = False
        meshData['nonManifoldEdges'].append([i, j])


    # step 2:
    # 2 ) check that the mesh don't have an hourglass shape a.k.a exist a
    #     vertex with 2-disk neighbourhood.
    myVertex = []
    for vertex in range(len(mesh.vertices)):
        myVertex.append(vertex)
        if np.isin(vertex, boundary):         # Something is wrong with this method in this case
            continue
        
        first = 0
        b = -1
        adjf = []
        # Get the face vertices adjacent to the vertex
        # take the first face that's contain's exactly two boundary vertices
        # and get it's vertices
        # [0, 1708]
        for face in mesh.vertex_faces[vertex]:
            if face != -1:
                faceVertices = np.where((face2Vertex[face, :] == 1).toarray())[1]
                adjf.append(faceVertices)
                verticesOnBoundary = np.sum(np.isin(faceVertices, boundary))
                if verticesOnBoundary == 2:
                    b = len(adjf) - 1
        adjf = np.array(adjf)
        
        # if we didn't find such face or we found one and v itself is not on
        # the boundary, then v is on a disk, else v is on a half disk.
        # b = 2
        if b == -1 or not np.isin(vertex, boundary):
            f = adjf[first]
            # take only the vertices on this face that are not v
            f = f[f != vertex]
            # take only the first one of those vertices
            f = f[0]
        else:
            first = b
            f = adjf[first]
            # take only the vertices on this face that are not v
            f = f[f != vertex]
            # take only the those vertices that are not on the boundary
            f = f[~np.isin(f, boundary)]
            # take only the first one of those vertices
            f = f[0]

        # find the only two possible faces that contain this vertex. aka the
        # face itself and the next neighbouring face.
        nextFace = np.where(np.any(adjf[:]==f, axis=1))
        # take the next face
        idx = np.where(nextFace[0] != first)[0][:]
        nextFace = [nextFace[0][i] for i in idx]
        # print('%d->%d',first,next)
        counter = 1
        while nextFace != first:
            curr = nextFace
            currf = f
            f = adjf[curr]
            f = f[0][numpy.logical_and((f[0] != vertex), (f[0] != currf))]
            nextFace = np.where(np.any(adjf==f, axis=1))[0]
            nextFace = nextFace[nextFace != curr]
            if nextFace.size == 0:
                nextFace = first
            else:
                pass
                #fprintf('->%d',next);
            counter += 1
        
        if counter != len(adjf):
            meshData['manifold'] = False
            meshData['nonManifoldVertices'].append(vertex)
        # print('\ncounter = %d, neighbours = %d\n\n',counter,size(adjf,2))

    # Depreciated
    # step 3:
    # 3 ) check that the every boundary vertex have only two neighbours on the
    #     boundary. aka the topology of the boundary is non disk shape.
    # edges = edges.tolil()
    # for b in boundary:
    #     myShape = edges[b,:].shape
    #     mySize = myShape[0] * myShape[1]
    #     flattened = edges[b,:].T.reshape((1, mySize))
    #     neigh = np.nonzero(flattened)
    #     neigh = np.nonzero(np.isin(neigh[1],boundary))
    #     if len(neigh[0]) > 2:
    #         meshData['nonManifoldBoundary'].append(b)
            
    return meshData

def HomeomorphismCheck(path=workingPath):
    '''
    This function serves two purposes. It will first check each surface in a
    particular folder to make sure that it is a manifold. It will
    simultaneously check to see if the surfaces are or are not discs.
    Nonmanifold surfaces and topology discrepancies will be reported.

    Function will return 1 if all manifolds are discs, 0 if all manifolds are
    not discs, and -1 if discrepancies are reported.

    Parameters
    ------------
    Names: list
      All the names of the meshes to be used

    Returns
    ----------
    result: dictionary
      Results of the check
        problemMeshes: list
          Names of the problem meshes
        badBoundaryMeshes: list
          Names of the meshes with bad boundaries
        discTopologyMeshes: list
          Names of the meshes with disc topology
        nonDiscTopologyMeshes: list
          Names of the meshes without disc topology
        numDiscs: int
          Number of discs
        numNonDiscs: int
          Number of non discs
        isDisc: int
          1 if there are only discs, 0 if there are non disc meshes
          -1 if there are problem meshes
        isMan: int
          -1 if there are problem meshes, 2 if not
    '''
    result = {}
    result['problemMeshes'] = []
    result['problemMeshPoints'] = []
    result['problemMeshProblems'] = []
    result['badBoundaryMeshes'] = []
    result['discTopologyMeshes'] = []
    result['nonDiscTopologyMeshes'] = []
    result['numDiscs'] = 0
    result['numNonDiscs'] = 0
    result['isDisc'] = -1
    result['isMan'] = 2
    problems = ['Problem with vertices', 'Problem with edges', 
            'Problem with vertices or edges', 'Unknown problem']

    Names = getNames(path)

    warnings.filterwarnings("ignore")
    
    for name in Names:
        print(name)
        mesh = trimesh.load(workingPath + 'RawOFF/' + name + '.off')
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()

        isManifoldResult = {}
        try:
          boundary, _ = FindOrientedBoundaries(mesh)
          boundaryList = list(chain.from_iterable(boundary))
          isManifoldResult = isManifold(mesh, boundaryList)
        except:
            isManifoldResult['manifold'] = False
            print("failed")

        # Check if manifold. If not, add to list. If yes, check boundary
        if not isManifoldResult['manifold']:
            result['problemMeshes'].append(name)
            ptCloud = []
            prob = 3
            if 'nonManifoldEdges' in isManifoldResult:
                if isManifoldResult['nonManifoldEdges']:
                  prob = 1
                  for edgeList in isManifoldResult['nonManifoldEdges']:
                      for edge in edgeList:
                          ptCloud.extend(edge)
                if isManifoldResult['nonManifoldVertices']:
                    if prob == 1:
                        prob = 2
                    else:
                        prob = 0
                    ptCloud.extend(isManifoldResult['nonManifoldVertices'])
                result['problemMeshPoints'].append(ptCloud)
                result['problemMeshProblems'].append(problems[prob])
        else:
            isManifoldResult['manifold'] = False
            if len(boundary) == 0:
                result['numNonDiscs'] += 1
                result['nonDiscTopologyMeshes'].append(name)
            elif len(boundary) == 1:
                result['numDiscs'] += 1
                result['discTopologyMeshes'].append(name)
            else:
                result['problemMeshes'].append(name)
                result['badBoundaryMeshes'].append(name)
                result['problemMeshPoints'].append(boundaryList)
                result['problemMeshProblems'].append('Problem with boundaries')

        
    warnings.filterwarnings("default")
    

    # In future this should be replaced with code that checks homology
    if result['badBoundaryMeshes']:
        print('ALERT: The manifold meshes do not have consistent topology. Please resolve before proceeding.')
        print('There are', str(result['numDiscs']), 'of those meshes with disc topology and',\
            str(result['numNonDiscs']), 'with non-disc topology.')
        print('There are', str(len(result['badBoundaryMeshes'])),\
            'meshes with boundary problems. Please check appropriately.')
        result['isDisc'] = -1
        result['isMan'] = -1

    if result['problemMeshes']:
        print('ALERT: The following meshes must be cleaned before proceeding:')
        for i in range(len(result['problemMeshes'])):
            print(result['problemMeshes'][i])
        
        result['isDisc'] = -1
        result['isMan'] = -1

        print('Displaying problem meshes')
        for i, name in enumerate(result['problemMeshes']):
            mesh = trimesh.load(workingPath + 'RawOFF/' + name + '.off')
            print(result['problemMeshProblems'][i])
            print(result['problemMeshPoints'][i])
            visualizePointsOnMesh(mesh, result['problemMeshPoints'][i], 3)
    
    if not (result['isMan'] == -1):
        if result['numNonDiscs'] > 0:
            result['isDisc'] = 0
        else:
            result['isDisc'] = 1
    
    # Print results to file    
    with open(workingPath + 'problemMeshes.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(result['problemMeshes'])
    
    with open(workingPath + 'discTopologyMeshes.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(result['discTopologyMeshes'])
    
    with open(workingPath + 'nonDiscTopologyMeshes.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(result['nonDiscTopologyMeshes'])
        
    return result


def Centralize(mesh, scale=None):
    '''
    Centralizes the mesh and gives unit surface area if given scale

    Parameters
    ------------
    mesh: trimesh mesh
      The mesh to be centralized
    scale: string
      will scale to unit surface area if passed anything
    depth: int
      How far from a vertex to search for a maximum from a vertex

    Returns
    ----------
    mesh: trimesh mesh
      New centralized mesh
    Center: float
      The new center of the mesh
    '''
    Center = np.mean(mesh.vertices, 0).reshape(1,3)
    foo = np.matlib.repmat(Center, len(mesh.vertices), 1)
    mesh.vertices -= foo

    if scale != None:
        mesh.vertices = mesh.vertices * np.sqrt(1 / mesh.area)

    return mesh, Center

def FindLocalMax(mesh, curvature, depth):
    '''
    Finds the local maximums of curvature given a depth

    Parameters
    ------------
    mesh: trimesh mesh
      The mesh to be computed
    curvature: (len(mesh.vertices),) float
      curavtures that correspond to each vertex
    depth: int
      How far from a vertex to search for a maximum from a vertex

    Returns
    ----------
    ind: (n,) float
      Vertex indices where there is a local maximum
    maxCurves: (n,) float
      Values associated with the local maximum
    '''

    num_vertices = len(mesh.vertices)
    sparseCurv = sparse.coo_matrix(np.diag(curvature))

    adjacency_matrix = nx.adjacency_matrix(mesh.vertex_adjacency_graph, 
                                           nodelist=range(num_vertices))
    power = (adjacency_matrix ** depth) + (adjacency_matrix ** (depth-1))   # Depth-adjacency
    AK = power + sparse.eye(num_vertices, num_vertices)   # Depth-adjacency + self
    AK[AK>0]=1

    AK_curves = AK.dot(sparseCurv)
    max_curves = np.max(AK_curves,axis=1).todense().T
    ind = np.equal(max_curves,np.array([curvature])).nonzero()

    return ind[1]

def ComputeCurvatureFeatures(mesh,CurvatureWindow=2,SmoothIterations=3):

    # Calculate base radius for pseudo discrete radius calculation
    base_radius = np.median(mesh.edges_unique_length)
    base_radius = np.max([CurvatureWindow*base_radius,np.max(mesh.edges_unique_length)+1e-8])

    # Calculate the curves for Gauss, mean, and DNE at each vertex
    gaussian_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, base_radius )
    mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, base_radius )

    DNE = (4 * np.square(mean_curvature)) - (2 * gaussian_curvature)

    #Compute Laplacian and smooth functions
    LB = trimesh.smoothing.laplacian_calculation(mesh)
    
    for si in range(SmoothIterations):
        gaussian_curvature,mean_curvature,DNE = LB.dot(gaussian_curvature),LB.dot(mean_curvature),LB.dot(DNE)
    return mesh, gaussian_curvature, mean_curvature, DNE


def visualizeMesh(mesh, points = None, values = None):
    '''
    Visualizes a mesh, with optional point cloud to highlight
    mesh vertices

    Parameters
    ------------
    mesh: trimesh mesh
      The mesh to be visualized
    points: (n) list of ints
      Mesh vertex indices that are to be highlighted
    values: (n) list of floats or ints
      Values to give an interpolated coloring to the mesh
    '''

    if points is not None:
        if values is not None:
            colors = trimesh.visual.color.interpolate(values, color_map='jet')
            pc2 = trimesh.points.PointCloud(mesh.vertices[points], colors=colors)
        else:
            pc2 = trimesh.points.PointCloud(mesh.vertices[points])
        trimesh.Scene([mesh, pc2]).show()
    else:
        trimesh.Scene(mesh).show()

#Compute conformal factors
def CalculateConformalFactor_Faces(meshA,meshB):
    triAreasA, triAreasB = meshA.area_faces, meshB.area_faces
    return np.divide(triAreasA,triAreasB)
def CalculateConformalFactor_Vertex(mesh,LambdaF,SmoothIterations=3):
    F2V = ComputeF2V(mesh)
    numFacesPerVertex = np.sum(F2V,axis=1)
    LambdaV = np.divide(F2V.dot(LambdaF),numFacesPerVertex)
    
    #Compute Laplacian and smooth
    LB = trimesh.smoothing.laplacian_calculation(mesh)

    for si in range(SmoothIterations):
        LambdaV = LB.dot(LambdaV)
    return LambdaV

def initSAMS(mesh, RawArea=None, RawCenter=None,
            gaussianCurvature=None, gaussMaxInds=None, 
            meanCurvature=None, meanMinInds=None, DNE=None, 
            DNEMaxInds=None, center=None, confFactorFaces=None,
            confFactorVertices=None, confMaxInds=None):

            
    sams = SAMSMesh(vertices=mesh.vertices, faces=mesh.faces, 
    face_normals=mesh.face_normals,vertex_normals=mesh.vertex_normals, 
    face_colors=mesh.face_attributes, vertex_colors=None, 
    face_attributes=None, vertex_attributes=mesh.vertex_attributes, 
    metadata=mesh.metadata, process=mesh.process, 
    validate=False, use_embree=True, initial_cache=None, visual=mesh.visual,
    RawArea=RawArea, RawCenter=RawCenter, gaussianCurvature=gaussianCurvature, 
    gaussMaxInds=gaussMaxInds, meanCurvature=meanCurvature, meanMinInds=meanMinInds, 
    DNE=DNE, DNEMaxInds=DNEMaxInds, center=center, confFactorFaces=confFactorFaces,
                confFactorVertices=confFactorVertices, confMaxInds=confMaxInds)

    return sams
    
def ComputeFeatures(path=workingPath):
    '''
    Computes the necessary features needed for calculations for all meshes

    Parameters
    ------------
    Names: list
      All the names of the meshes to be used


    Returns
    ----------
    In future will either store features in class or some other file
    '''
    output_path = path + 'ProcessedSAMS/'
    touch(output_path)
    Names = getNames(path)
    print('Loading Meshes')
    meshes = []
    for name in Names:
        print(name)
        mesh = trimesh.load(workingPath + 'RawOFF/' + name + '.off')
        RawArea = mesh.area
        [mesh, RawCenter] = Centralize(mesh, 'ScaleArea')

        
        # Curvature Computation and Local Extrema
        mesh, gaussianCurvature, meanCurvature, DNE = ComputeCurvatureFeatures(mesh,CurvatureWindow,SmoothIterations)
        gaussMaxInds = FindLocalMax(mesh, gaussianCurvature, GaussMaxLocalWidth)
        meanMinInds = FindLocalMax(mesh,-np.abs(meanCurvature),MeanMinLocalWidth)
        DNEMaxInds = FindLocalMax(mesh,DNE,DNEMaxLocalWidth)
        
        #Conformal Factor Computation and Local Extrema
        # uniV = spherical_conformal_map(mesh.vertices,mesh.faces)
        # uniMesh = trimesh.Trimesh(vertices = uniV,faces = mesh.faces)
        # confFactorFaces = CalculateConformalFactor_Faces(mesh,uniMesh)
        # confFactorVertices = CalculateConformalFactor_Vertex(mesh,confFactorFaces,SmoothIterations=SmoothIterations)
        # confMaxInds = FindLocalMax(mesh,confFactorVertices,ConfMaxLocalWidth)

        sams = initSAMS(mesh, RawArea, RawCenter, gaussianCurvature, gaussMaxInds, meanCurvature,
        meanMinInds, DNE, DNEMaxInds)#, confFactorFaces, confFactorVertices,
        #confMaxInds)

        meshes.append(sams)

        with open(output_path + name, 'wb') as outfile:
            pickle.dump(sams, outfile)

        # with open(output_path + name, 'rb') as sams_file:
        #     saveSams = pickle.load(sams_file)
    
    return meshes
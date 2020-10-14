import os
import sys
import csv
import json

import numpy as np
import networkx as nx
import numpy.matlib
import numpy.linalg
import trimesh
import copy
import pickle

from scipy import sparse, spatial
from numpy import genfromtxt
from sklearn.neighbors import NearestNeighbors
from Setup.initialize import workingPath
from Setup.MappingSetup import featureMap, maxDistTol
from touch import touch

def saveJson(path, data):
    with open(workingPath + path, 'w') as outfile:
        json.dump(data, outfile)

def DFS(M, thresh, startInd, finalInd, maxDepth):
    '''
    Using DFS, finds all paths starting at startInd and ending at 
    finalInd that have a path weight of at least thresh

    Parameters
    ------------
    M: (n x n) sparse lil adjacency matrix
      Adjacency matrix of all the weights
    thresh: float or int
      Threshold of the weighted path
    startInd: int
      Specifies the starting index of the path
    finalInd: int
      Specifies the final index of the path
    maxDepth: int
      Maximum search depth of the search


    Returns
    ----------
    paths: list of list of ints
      list of paths from startInd to finalInd with weight at least thresh
    '''
    paths = []
    def DepthFirstSearchPlotting_12(M, thresh, startInd, finalInd, maxDepth, curPath, curDepth=1, weight=1):
        
        curInd = curPath[-1]

        _, nextVerts = M[curInd, :].nonzero()

        for vert in nextVerts:
            if (curDepth == maxDepth) and (vert != finalInd):
                continue
            if vert in curPath:
                continue

            nextWeight = weight * M[curInd, vert]
            if (nextWeight < thresh):
                continue
            
            nextPath = copy.deepcopy(curPath)
            nextPath.append(vert)
            if vert == finalInd:
                paths.append(nextPath)
            else:
                DepthFirstSearchPlotting_12(M, thresh, startInd, finalInd, maxDepth, nextPath, curDepth + 1, weight)

    curPath = [startInd]

    DepthFirstSearchPlotting_12(M, thresh, startInd, finalInd, maxDepth, curPath)

    return paths

def ComputeDirectedFlows(dists, sourceInds, sinkInds):
    '''
    TODO finish comments
    Returns the directed flow matrix required for improving pairwise
    correspondence

    Parameters
    ------------
    dists: (m x n) numpy ndarray of floats
      Distribution of GP
    sourceInds: (m) list or range of indices
      Specifies the source indices
    sinkInds: list or range of indices
      Specifies the sink indices


    Returns
    ----------
    Flows: 
      list of sparse matrices size m x n with each entry denoting the directed 
      adjacency matrix of flows from i to j
    '''

    m, n = np.shape(dists)
    dists = 0.5 * dists + 0.5 * dists.T         # symmetrize as sanity check
    Flows = [[None] * len(sinkInds) for i in range(len(sourceInds))]

    for i in range(len(sourceInds)):
        for j in range(len(sinkInds)):
            print(i, ',', j)
            Flows[i][j] = sparse.lil_matrix((m,n))
            if sourceInds[i] != sinkInds[j]:
                dummy = sparse.lil_matrix((m,n), dtype='int')
                d_i = sparse.csgraph.shortest_path(sparse.coo_matrix(dists), indices = sourceInds[i])
                d_j = sparse.csgraph.shortest_path(sparse.coo_matrix(dists), indices = sinkInds[j])
                for k in range(m):
                    for q in range(n):
                        if d_i[k] < d_i[q]:
                            if d_j[k] > d_j[q]:
                                dummy[k, q] = 1
                Flows[i][j] = dummy
            else:
                Flows[i][j][sourceInds[i], sinkInds[j]] = 1
    return Flows

def MappingSetupAndFlowExtraction(Names, Flags):
    '''
    Computes or loads GPDists and Flows of all meshes
    '''

    input_path = workingPath + 'ProcessedSAMS/'
    meshList = []
    # GPLmkList = []    # TODO implement GPLmk when GPLmk is done
    PtCloudList = []
    for name in Names:
        # mesh = trimesh.load(workingPath + 'ProcessedTRI/' + name + '.off')
        with open(input_path + name, 'rb') as sams_file:
            mesh = pickle.load(sams_file)
        meshList.append(mesh)
        # GPLmkList{i} = G.Aux.GPLmkInds    # TODO implement GPLmk when GPLmk is done
        vertices = mesh.vertices
        # centralize point clouds
        vertices = vertices - np.matlib.repmat(np.mean(vertices, axis=0), len(vertices), 1)
        vertices = vertices / np.linalg.norm(vertices)
        pointCloud = trimesh.points.PointCloud(vertices)
        PtCloudList.append(pointCloud)
    
    # First step: load distances if they exist or compute if needed

    if os.path.exists(workingPath + 'GPDists.npy'):
        GPDists = np.load(workingPath + 'GPDists.npy')
    else:
        print('Distances not found, computing...')
        GPDists = np.zeros((len(Names), len(Names)))
        for i in range(len(Names)):
            # procMaps = np.zeros((len(Names), len(Names)))
            # dummy = np.zeros((len(Names)))
            curCloud = PtCloudList[i]
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(curCloud.vertices)
            # TODO implement muliprocessing, parfor in matlab
            for j in range(len(Names)):
                if i != j:
                    # [P,procDists(i,j),~] = linassign(ones(size(GPPtClouds{i},2),size(GPPtClouds{i},2)),D);
                    # Get map from permutation
                    distances, _ = nbrs.kneighbors(PtCloudList[j].vertices)
                    GPDists[i,j] = np.mean(distances)
        GPDists = (GPDists+GPDists.T) / 2
        np.save(workingPath + 'GPDists.npy', GPDists)
            
    Flags['hasDists'] = True
    saveJson("Flags.json", Flags)

    # Flows
    if not ('hasFlows' in Flags):
        frInd = np.where(sum(GPDists ** 2)==min(sum(GPDists ** 2)))
        frInd = [frInd[0][0]]
        Flows = ComputeDirectedFlows(GPDists, range(len(GPDists)), frInd)
        Flags['hasFlows'] = True
        saveJson("Flags.json", Flags)
        np.save(workingPath + 'Flows', Flows)
                
    else:
        frInd = np.where(sum(GPDists ** 2)==min(sum(GPDists ** 2)))       # This is needed later
        print('Flows to Frechet Mean already computed, loading...')
        Flows = np.load(workingPath + 'Flows.npy', allow_pickle=True)

    return Flags

def getDistance(mesh, start, end):
    '''
    Finds the distance between two points on a mesh

    Parameters
    ------------
    mesh: trimesh mesh
      The mesh for 
    start: int
      Starting vertex index
    end: int
      Ending vertex index


    Returns
    ----------
    distance: float
      The distance between the start and end vertex indices
    '''


    edges = mesh.edges_unique
    length = mesh.edges_unique_length
    
    g = nx.from_edgelist([(e[0], e[1], {'length': L})
           for e, L in zip(edges, length)])
    
    distance = nx.shortest_path_length(g,
            source=start,
            target=end,
            weight='length')
    
    return distance
        
        
def ComputeFeatureMatching(Names, Flags):
    '''
    # TODO
    Finds features on each mesh that correspond to the frech mean features

    Parameters
    ------------
    Names: list of strings
      Names of the meshes
    Flags: dictionary
      Dictionary of the flags of steps executed


    Returns
    ----------
    distance: float
      The distance between the start and end vertex indices
    '''

    meshList = []
    input_path = workingPath + 'ProcessedSAMS/'
    for name in Names:
        with open(input_path + name, 'rb') as sams_file:
            mesh = pickle.load(sams_file)
        meshList.append(mesh)

    GPDists = np.load(workingPath + 'GPDists.npy')
    sumGPDists = np.sum(GPDists ** 2, axis=0)
    minGPDists = np.min(sumGPDists)
    frechMean = np.where(minGPDists == sumGPDists)
    frechMean = frechMean[0][0]

    featureList = []
    if featureMap == 'Conf':
        for mesh in meshList:
            featureList.append(mesh.confMaxInds)
    elif featureMap == 'Gauss':
        for mesh in meshList:
            featureList.append(mesh.gaussMaxInds)
    elif featureMap == 'Mean':
        for mesh in meshList:
            featureList.append(mesh.meanMinInds)
    elif featureMap == 'DNE':
        for mesh in meshList:
            featureList.append(mesh.DNEMaxInds)
    else:
        raise Exception("Unknown featureMap variable. Please only use Conf, Gauss \
            Mean, or DNE for the featureMap variable.")

    featureMatchesPairs = [[] for i in Names]
    meshFrech = meshList[frechMean]
    for i in range(len(Names)):
        print('Computing features for', Names[i])
        if i != frechMean:
            mesh = meshList[i]

            meshFrechFeature = meshFrech.vertices[featureList[frechMean], :]
            meshOtherFeature = mesh.vertices[featureList[i], :]

            nbrs = NearestNeighbors(n_neighbors=1, 
                                    algorithm='ball_tree').fit(meshFrechFeature)
            _, map_12 = nbrs.kneighbors(meshOtherFeature)

            nbrs = NearestNeighbors(n_neighbors=1, 
                                    algorithm='ball_tree').fit(meshOtherFeature)
            _, map_21 = nbrs.kneighbors(meshFrechFeature)

            for j in range(len(map_12)):
                
                # You did this because you wanted to restrict analysis to GP
                # landmarks, so you projected onto GP landmarks because you were
                # concerned that there would be overlap

                if map_21[map_12[j]] == j:
                    meshOther = [mesh.vertices[featureList[i][j]]]
                    nbrs = NearestNeighbors(n_neighbors=1, 
                                            algorithm='ball_tree').fit(meshFrech.vertices)
                    _, lmk12 = nbrs.kneighbors(meshOther)
                    

                    meshFrechPts = meshList[frechMean].vertices[featureList[frechMean][map_12[j]], :]
                    nbrs = NearestNeighbors(n_neighbors=1, 
                                            algorithm='ball_tree').fit(mesh.vertices)
                    _, lmk21 = nbrs.kneighbors(meshFrechPts)
                    
                    
                    start12 = lmk12[0, 0]
                    end12 = featureList[frechMean][map_12[j][0]]
                    dist12 = getDistance(meshFrech, start12, end12)
                    
                    start21 = lmk21[0, 0]
                    end21 = featureList[i][j]
                    dist21 = getDistance(mesh, start21, end21)
                    
                    
                    if dist12 < maxDistTol and dist21 < maxDistTol:
                        featureMatchesPairs[i].append([featureList[i][j], 
                                                       featureList[frechMean][map_12[j]][0]])


    Flags['featureMappings'] = True
    saveJson('Flags.json', Flags)
    output = workingPath + 'MappingData'
    touch(output)
    np.save(output + '/FeatureMatches', featureMatchesPairs)

    return Flags
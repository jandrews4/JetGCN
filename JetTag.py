import numpy as np
import pandas as pd
from operator import truth
import pandas as pd
import numpy as np
import awkward as ak
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import uproot
import torch
from tqdm import tqdm
import timeit
import os
# take ROOT file and convert to an awkward array
def fileToAwk(path):
    file = uproot.open(path)
    tree = file['tree']
    
    awk = tree.arrays(tree.keys())
    return awk

input_features = ["part_px", "part_py", "part_pz", "part_energy",
                  "part_deta", "part_dphi", "part_d0val", "part_d0err", 
                  "part_dzval", "part_dzerr", "part_isChargedHadron", "part_isNeutralHadron", 
                  "part_isPhoton", "part_isElectron", "part_isMuon" ] # features used to train the model

 
# take AWK dict and convert to a point cloud
def awkToPointCloud(awkDict, input_features):
    available_features = awkDict # all features

    featureVector = []
    for jet in tqdm(range(len(awkDict)), total=len(awkDict)):
        currJet = awkDict[jet][input_features]
        pT = np.array(np.sqrt(currJet['part_px'] ** 2 + currJet['part_py'] ** 2))
        # creates numpy array to represent the 4 momenta of all particles in a jet
        currJet = np.column_stack((np.array(currJet['part_px']), np.array(currJet['part_py']), 
                                   np.array(currJet['part_pz']), np.array(currJet['part_energy']), pT
                                   , np.array(currJet['part_deta']), np.array(currJet['part_dphi']), 
                                   np.array(currJet["part_d0val"]), np.array(currJet["part_d0err"]), 
                                   np.array(currJet["part_dzval"]), np.array(currJet["part_dzerr"]), 
                                   np.array(currJet["part_isChargedHadron"]), np.array(currJet["part_isNeutralHadron"]), 
                                   np.array(currJet["part_isPhoton"]), np.array(currJet["part_isElectron"]), 
                                   np.array(currJet["part_isMuon"])))
        
        featureVector.append(currJet)
    return np.asarray(featureVector, dtype="object")
from scipy.spatial import cKDTree

#take point cloud and build KNN graph
def buildKNNGraph(points, k):
    
    # Compute k-nearest neighbors
    tree = cKDTree(points)
    dists, indices = tree.query(points, k+1)  # +1 to exclude self
    
    # Build adjacency matrix
    num_points = len(points)
    adj_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in indices[i, 1:]:  # exclude self
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
    return adj_matrix

# take adjacency matrix and turn it into a DGL graph
def adjacencyToDGL(adj_matrix):
    adj_matrix = sp.coo_matrix(adj_matrix)
    g_dgl = dgl.from_scipy(adj_matrix)
        
    return g_dgl
# process all jetTypes
Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
Vector = ['WToQQ', 'ZToQQ']
Top = ['TTBar', 'TTBarLep']
QCD = ['ZJetsToNuNu']
Emitter = ['Emitter-Vector', 'Emitter-Top', 'Emitter-Higgs', 'Emitter-QCD']
allJets = Higgs + Vector + Top + QCD

#for jetType in allJets:
    #fileToGraph(jetType)

#allGraphs = groupToGraph(Higgs, "Emitter-Higgs")
#with open(f'/data/train/Multi Level Jet Tagging/Emitter-Higgs.pkl', 'wb') as f:
    #pickle.dump(allGraphs, f)
    
print("DONE")
import scipy.sparse as sp
import dgl
import pickle

# wrap the functionality of fileToAwk and awkToPointCloud in a function to return a point cloud numpy array
def fileToPointCloudArray(jetType, input_features):
    filepath = f'data/train/{jetType}' # original root file
    savepath = f'data/train/{jetType}'.replace('.root' , '')+'.npy' # save file
    awk = fileToAwk(filepath)
    nparr = awkToPointCloud(awk, input_features)
    
    return nparr

# wrap the functionality of fileToPointCloudArray and the 
def fileToGraph(jetType, k=3, save=True):
    print(f'Starting processing on {jetType} jets')
    pointCloudArr = fileToPointCloudArray(jetType, input_features)
    
    saveFilePath = f'data/train/{jetType}.pkl'
    
    savedGraphs = []
    for idx, pointCloud in tqdm(enumerate(pointCloudArr), leave=False, total=len(pointCloudArr)):
    
            adj_matrix = buildKNNGraph(pointCloud, k)
            graph = adjacencyToDGL(adj_matrix)
            
            graph.ndata['feat'] = torch.tensor(pointCloud, dtype=torch.float32)
            
            savedGraphs.append(graph)
            
            del adj_matrix, graph
            
    
    if save:
        with open(saveFilePath, 'wb') as f:
            pickle.dump(savedGraphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        del pointCloudArr, savedGraphs
        
    print(f'Graphs for {jetType} processing complete!')
        
    if 3 >2:
        return savedGraphs

def groupToGraph(jetTypeList, groupName):
    allGraphs = []
    for jetType in jetTypeList:
        allGraphs += fileToGraph(jetType, save=False)
    
    saveFilePath = f'data/train/{groupName}.pkl' 
    return allGraphs
# import required module
import os
# assign directory
directory = 'G:/PCN-Jet-Tagging-master/data/train'

# iterate over files in
# that directory
for filename in os.listdir(directory):
	fileToGraph(filename)
import uproot
import numpy as np
import networkx as nx

def convert_jetclass_to_graph(root_file):
    # Open the ROOT file
    file = uproot.open(root_file)
    
    # Access the jets array from the tree
    jets_array = file["tree;1"].arrays(library="np")
    
    # Create a list to hold the graphs
    graphs = []
    print(jets_array.keys())
    # Identify events based on consecutive jet indices
    event_starts = np.where(jets_array['jet_id'][1:] < jets_array['jet_id'][:-1])[0] + 1
    event_starts = np.insert(event_starts, 0, 0)
    event_starts = np.append(event_starts, len(jets_array))
    
    # Loop over each event
    for i in range(len(event_starts) - 1):
        event_indices = np.arange(event_starts[i], event_starts[i+1])
        
        # Create a new graph
        G = nx.Graph()
        
        # Add nodes (jets) to the graph
        for idx in event_indices:
            node_attrs = {
                "pt": jets_array['pt'][idx],
                "eta": jets_array['eta'][idx],
                "phi": jets_array['phi'][idx],
                "mass": jets_array['mass'][idx],
                "jet_id": jets_array['jet_id'][idx]
            }
            G.add_node(jets_array['jet_id'][idx], **node_attrs)
        
        # Add edges (connections between jets) based on some criteria
        # You can define your own criteria here
        # For example, connecting all jets within a certain delta-eta and delta-phi
        for idx1 in range(len(event_indices)):
            for idx2 in range(idx1 + 1, len(event_indices)):
                jet_id1 = jets_array['jet_id'][event_indices[idx1]]
                jet_id2 = jets_array['jet_id'][event_indices[idx2]]
                d_eta = jets_array['eta'][event_indices[idx1]] - jets_array['eta'][event_indices[idx2]]
                d_phi = np.abs(jets_array['phi'][event_indices[idx1]] - jets_array['phi'][event_indices[idx2]])
                if d_phi > np.pi:
                    d_phi = 2 * np.pi - d_phi
                distance = np.sqrt(d_eta**2 + d_phi**2)
                
                # Threshold distance for creating an edge
                if distance < 0.4:  # Example threshold, you can adjust as needed
                    G.add_edge(jet_id1, jet_id2)
        
        # Add the graph to the list
        graphs.append(G)
    
    return graphs

# Example usage
root_file_path = "/media/jacob/maxone/PCN-Jet-Tagging-master/data/train/WToQQ_034.root"
graphs = convert_jetclass_to_graph(root_file_path)

# Now you have a list of NetworkX graphs where each graph represents an event
# You can further process or use these graphs for your Graph Neural Network

import uproot

# Open the ROOT file
file = uproot.open("/media/jacob/maxone/PCN-Jet-Tagging-master/data/train/WToQQ_034.root")

# Print available keys
print(file.keys())

import torch
import pandas as pd
import os
import glob
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler
import zipfile
import pickle
import torch.nn as nn

OP_TO_IDX = {'fsto': 0, 'rf': 1, 'rs': 2, 'rw': 3, 'b': 4, 'fres': 5, 'st': 6, 'f': 7, 'rwz':8, 'rfz':9}
 
def tokenize_recipe(recipe_str, max_len=20):
    ops = recipe_str.strip().split(',')
    tokens = [OP_TO_IDX[o] for o in ops]
    tokens += [0] * (max_len - len(tokens))
    return torch.tensor(tokens[:max_len])


def load_graph_embeddings(pt_folder):
    pt_files = glob.glob(os.path.join(pt_folder, "*.zip"))
    data_dict = {} 
    for file in pt_files:
        name = os.path.basename(file).replace(".pt.zip", "")  # Extract design name
        extract_path = f"/tmp/{name}"                         # Unique temp folder
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(file, 'r') as zipf:
            zipf.extractall(extract_path)
        
        # Find the actual .pt file inside the zip
        pt_file = glob.glob(os.path.join(extract_path, "*.pt"))[0]
        data = torch.load(pt_file, weights_only=False)
        
        data_dict[name] = data
    
    return data_dict

class GraphFeatureExtractor(nn.Module):
    def __init__(self, num_node_types, embedding_dim=4):
        super().__init__()
        self.node_type_embed = nn.Embedding(num_node_types, embedding_dim)
        self.numeric_proj = nn.Linear(1, embedding_dim)

    def forward(self, node_type, num_inverted_predecessors):
        if not isinstance(node_type, torch.Tensor):
            node_type = torch.tensor(node_type, dtype=torch.long)
        if not isinstance(num_inverted_predecessors, torch.Tensor):
            num_inverted_predecessors = torch.tensor(num_inverted_predecessors, dtype=torch.float32)

        num_inverted_predecessors = num_inverted_predecessors.view(-1, 1)
        numeric_embedded = self.numeric_proj(num_inverted_predecessors)
        node_type_embedded = self.node_type_embed(node_type)

        node_features = torch.cat([node_type_embedded, numeric_embedded], dim=-1)
        return node_features

num_node_types = 3  # Set this based on the number of node types in your graph
feature_extractor = GraphFeatureExtractor(num_node_types)

# Function to create dataset using graph and recipe data
def create_dataset_with_features(graph_folder, recipe_folder, label_folder):
    X = []
    Y = []
    graph_data = load_graph_embeddings(graph_folder)  # This loads your graph embeddings
    for design, g in graph_data.items():
        recipe_path = os.path.join(recipe_folder, f"{design}.csv")
        label_path = os.path.join(label_folder, f"{design}.csv")
        recipes = pd.read_csv(recipe_path)['recipe']
        labels = pd.read_csv(label_path)['delay']

        # Extract node features using the GraphFeatureExtractor
        node_types = g.node_type  # Assuming these are present
        num_inverted_predecessors = g.num_inverted_predecessors  # Assuming these are present

        # Ensure num_inverted_predecessors is numeric
        num_inverted_predecessors = [float(val) for val in num_inverted_predecessors]  # Convert to float

        # Extract features
        node_features = feature_extractor(node_types, num_inverted_predecessors)
        # Construct the graph data with the feature matrix `x`
        graph_data_with_features = Data(x=node_features, edge_index=g.edge_index)

        for rec, y in zip(recipes, labels):
            X.append((graph_data_with_features, tokenize_recipe(rec)))  # Recipe embedding will be added
            Y.append(torch.tensor(y, dtype=torch.float))

    print("Length of X:", len(X))
    print("Length of Y:", len(Y))
    return X, Y

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm

class EnhancedGraphEncoder(nn.Module):
    def __init__(self, in_channels=8, hidden_dim=128):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x1 = self.relu(self.bn1(self.conv1(x, edge_index)))
        x2 = self.relu(self.bn2(self.conv2(x1, edge_index)))
        x = x1 + x2  # Residual connection
        x = self.dropout(x)
        return global_mean_pool(x, batch)

# import torch
# import torch.nn as nn
# from torch_geometric.nn import GCNConv, global_mean_pool

# class GraphEncoder(nn.Module):
#     def __init__(self, in_channels=5, hidden_dim=128):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x, edge_index, batch):
#         x = self.relu(self.conv1(x, edge_index))
#         x = self.relu(self.conv2(x, edge_index))
#         x = global_mean_pool(x, batch)
#         return x
    
# import torch
# import torch.nn as nn
# from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

# class GraphEncoder(nn.Module):
#     def __init__(self, in_channels=5, hidden_dim=128, dropout=0.2):
#         super(GraphEncoder, self).__init__()
        
#         self.conv1 = GCNConv(in_channels, hidden_dim)
#         self.bn1 = BatchNorm(hidden_dim)

#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.bn2 = BatchNorm(hidden_dim)

#         self.conv3 = GCNConv(hidden_dim, hidden_dim)
#         self.bn3 = BatchNorm(hidden_dim)

#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, edge_index, batch):
#         # First layer (no residual needed)
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)

#         # Second layer with residual
#         x_res = x
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = self.relu(x + x_res)  # Add residual
#         x = self.dropout(x)

#         # Third layer with residual
#         x_res = x
#         x = self.conv3(x, edge_index)
#         x = self.bn3(x)
#         x = self.relu(x + x_res)  # Add residual
#         x = self.dropout(x)

#         # Global pooling for graph representation
#         x = global_mean_pool(x, batch)
#         return x


# import torch
# import torch.nn as nn
# from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

# class GraphEncoder(nn.Module):
#     def __init__(self, in_channels=5, hidden_dim=128, dropout=0.2):
#         super(GraphEncoder, self).__init__()
        
#         self.conv1 = GCNConv(in_channels, hidden_dim)
#         self.bn1 = BatchNorm(hidden_dim)

#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.bn2 = BatchNorm(hidden_dim)

#         self.conv3 = GCNConv(hidden_dim, hidden_dim)
#         self.bn3 = BatchNorm(hidden_dim)

#         # Gates for residual connections (1 for each residual connection)
#         self.gate2 = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 1),
#             nn.Sigmoid()
#         )
#         self.gate3 = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 1),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, edge_index, batch):
#         # Layer 1 (no residual)
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)

#         # Layer 2 with gated residual
#         x_res = x
#         x2 = self.conv2(x, edge_index)
#         x2 = self.bn2(x2)

#         gate_input = torch.cat([x2, x_res], dim=-1)
#         gate = self.gate2(gate_input)
#         x = gate * x2 + (1 - gate) * x_res
#         x = self.relu(x)
#         x = self.dropout(x)

#         # Layer 3 with gated residual
#         x_res = x
#         x3 = self.conv3(x, edge_index)
#         x3 = self.bn3(x3)

#         gate_input = torch.cat([x3, x_res], dim=-1)
#         gate = self.gate3(gate_input)
#         x = gate * x3 + (1 - gate) * x_res
#         x = self.relu(x)
#         x = self.dropout(x)

#         # Global pooling
#         x = global_mean_pool(x, batch)
#         return x



# import torch
# import torch.nn as nn
# from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm

# class GraphEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_dim=128, dropout=0.3):
#         super().__init__()
#         self.project_in = nn.Linear(in_channels, hidden_dim)  # To align dim for residual

#         self.conv1 = SAGEConv(hidden_dim, hidden_dim)
#         self.bn1 = BatchNorm(hidden_dim)

#         self.conv2 = SAGEConv(hidden_dim, hidden_dim)
#         self.bn2 = BatchNorm(hidden_dim)

#         self.conv3 = SAGEConv(hidden_dim, hidden_dim)
#         self.bn3 = BatchNorm(hidden_dim)

#         self.activation = nn.LeakyReLU(0.1)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, edge_index, batch):
#         x = self.project_in(x)

#         # First layer with residual
#         res = x
#         x = self.conv1(x, edge_index)
#         x = self.activation(x + res)
#         x = self.dropout(x)

#         # Second layer with residual
#         res = x
#         x = self.conv2(x, edge_index)
#         x = self.activation(x + res)
#         x = self.dropout(x)

#         # Graph-level embedding
#         x = global_mean_pool(x, batch)
#         return x

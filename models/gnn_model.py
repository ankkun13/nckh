"""
GNN Model — Driver Gene Identification
=======================================
Multi-graph GNN using PyTorch Geometric's GCNConv or GATConv.
Replaces MODCAN's custom MGCN with PyG-based architecture.

Features:
  - 2-3 layers (avoids over-smoothing)
  - Residual/skip connections (per SGCD paper insights)
  - Multi-graph input (one per patient cluster slice)
  - Full CUDA support with dynamic device selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class DriverGeneGNN(nn.Module):
    """
    Graph Neural Network for cancer driver gene prediction.
    
    Architecture:
      Input → [GCNConv/GATConv × (n_layers-1)] → Linear → Output(2)
      With residual connections between layers to prevent over-smoothing.
    
    Supports multi-graph input: multiple adjacency slices (one per cluster)
    are processed and concatenated before final classification.
    """
    
    def __init__(self, in_channels: int, hidden_channels: list,
                 num_classes: int = 2, n_graphs: int = 1,
                 conv_type: str = 'GCN', dropout: float = 0.5,
                 use_residual: bool = True):
        """
        Args:
            in_channels: Number of input features per gene.
            hidden_channels: List of hidden dimensions [64, 128].
            num_classes: Number of output classes (2: driver/non-driver).
            n_graphs: Number of graph slices (patient clusters).
            conv_type: 'GCN' or 'GAT'.
            dropout: Dropout probability.
            use_residual: Whether to use residual connections.
        """
        super(DriverGeneGNN, self).__init__()
        
        self.n_graphs = n_graphs
        self.conv_type = conv_type
        self.dropout = dropout
        self.use_residual = use_residual
        self.hidden_channels = hidden_channels
        
        # Build graph convolution layers for each slice
        self.conv_layers = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        
        # Per-slice convolution pipeline
        for s in range(n_graphs):
            layers = nn.ModuleList()
            res_projs = nn.ModuleList()
            
            prev_dim = in_channels
            for i, h_dim in enumerate(hidden_channels):
                if conv_type == 'GCN':
                    layers.append(GCNConv(prev_dim, h_dim, add_self_loops=False))
                elif conv_type == 'GAT':
                    # GAT with 4 attention heads, concatenated
                    heads = 4 if i < len(hidden_channels) - 1 else 1
                    layers.append(GATConv(prev_dim, h_dim // heads,
                                          heads=heads, dropout=dropout))
                else:
                    raise ValueError(f"Unknown conv_type: {conv_type}")
                
                # Residual projection if dimensions don't match
                if use_residual and prev_dim != h_dim:
                    res_projs.append(nn.Linear(prev_dim, h_dim, bias=False))
                else:
                    res_projs.append(nn.Identity())
                
                prev_dim = h_dim
            
            self.conv_layers.append(layers)
            self.residual_projs.append(res_projs)
        
        # Batch normalization per layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(h_dim, eps=1e-5) for h_dim in hidden_channels
        ])
        
        # Classification head
        total_dim = hidden_channels[-1] * n_graphs
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor, edge_indices: list,
                edge_weights: list = None) -> tuple:
        """
        Forward pass with multi-graph input.
        
        Args:
            x: Node feature matrix (n_nodes x in_channels).
            edge_indices: List of edge_index tensors (one per graph slice).
            edge_weights: Optional list of edge weight tensors.
        
        Returns:
            Tuple of (raw logits, softmax probabilities).
        """
        slice_outputs = []
        
        for s in range(min(self.n_graphs, len(edge_indices))):
            h = x
            edge_index = edge_indices[s]
            edge_weight = edge_weights[s] if edge_weights else None
            
            for layer_idx, conv in enumerate(self.conv_layers[s]):
                # Graph convolution
                if edge_weight is not None and self.conv_type == 'GCN':
                    h_new = conv(h, edge_index, edge_weight=edge_weight)
                else:
                    h_new = conv(h, edge_index)
                
                # Batch normalization
                h_new = self.batch_norms[layer_idx](h_new)
                
                # ReLU activation (except last layer)
                if layer_idx < len(self.hidden_channels) - 1:
                    h_new = F.relu(h_new)
                    h_new = F.dropout(h_new, p=self.dropout, training=self.training)
                
                # Residual connection
                if self.use_residual:
                    h_res = self.residual_projs[s][layer_idx](h)
                    h_new = h_new + h_res
                
                # ✅ SAFETY: Prevent values from exploding in deep layers or mixed precision
                h_new = torch.nan_to_num(h_new, nan=0.0, posinf=100.0, neginf=-100.0)
                
                h = h_new
            
            slice_outputs.append(h)
        
        # Concatenate outputs from all graph slices
        if len(slice_outputs) > 1:
            combined = torch.cat(slice_outputs, dim=1)
        else:
            combined = slice_outputs[0]
        
        # Classification
        logits = self.classifier(combined)
        probs = F.softmax(logits, dim=1)
        
        return logits, probs


class MGCNLegacy(nn.Module):
    """
    Legacy MGCN architecture for compatibility with original MODCAN.
    Uses custom hypergraph convolution with einsum operations.
    
    This is provided for comparison; DriverGeneGNN is the recommended model.
    """
    
    def __init__(self, n_input: int, n_edge: int, hidden_dims: list,
                 dropout: float = 0.5):
        super(MGCNLegacy, self).__init__()
        
        self.n_input = n_input
        self.n_edge = n_edge
        self.hidden_dims = hidden_dims
        
        self.layer_0 = nn.Linear(n_input * n_edge, hidden_dims[0], bias=True)
        self.layer_1 = nn.Linear(hidden_dims[0] * n_edge, hidden_dims[1], bias=True)
        self.layer_2 = nn.Linear(hidden_dims[1], 2, bias=True)
        self.dropout = nn.Dropout(p=dropout)
    
    def _calculate_slice_tensor(self, data: torch.Tensor) -> torch.Tensor:
        slice_num = data.shape[2]
        mgcn_x = data[:, :, 0]
        for s in range(1, slice_num):
            mgcn_x = torch.cat((mgcn_x, data[:, :, s]), dim=1)
        return mgcn_x
    
    def forward(self, feature: torch.Tensor,
                hp_graph: torch.Tensor) -> tuple:
        mgcn_x0 = self._calculate_slice_tensor(
            torch.einsum("nij,ik->nkj", hp_graph, feature)
        )
        mgcn_x0 = self.dropout(F.relu(self.layer_0(mgcn_x0)))
        mgcn_x1 = self._calculate_slice_tensor(
            torch.einsum("nij,ik->nkj", hp_graph, mgcn_x0)
        )
        mgcn_x1 = F.relu(self.layer_1(mgcn_x1))
        output = self.layer_2(mgcn_x1)
        return output, F.softmax(output, dim=1)

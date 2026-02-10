"""
Simple GCN layers as an alternative to the scattering/wavelet transform.
This allows for ablation studies comparing structured wavelets vs. standard GCN message passing.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


class GCN_Layer(nn.Module):
    """
    Simple GCN-based feature extraction as an alternative to Scatter_layer.
    Uses multiple GCN layers followed by MLPs to produce outputs in the same format
    as the scattering transform for drop-in replacement.
    
    Args:
        in_channels: Input feature dimension
        max_graph_size: Maximum number of nodes (for batching)
        num_layers: Number of GCN layers (default: 4, to match scattering depth)
        hidden_channels: Hidden dimension for GCN layers
    """
    
    def __init__(self, in_channels, max_graph_size, num_layers=4, hidden_channels=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.max_graph_size = max_graph_size
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels or in_channels
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.gcn_layers.append(GCNConv(in_channels, self.hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNConv(self.hidden_channels, self.hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))
        
        # Last layer
        self.gcn_layers.append(GCNConv(self.hidden_channels, in_channels))
        self.batch_norms.append(nn.BatchNorm1d(in_channels))
        
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
        
        # MLP to process each layer's output (similar to scatter MLPs)
        self.layer_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, in_channels),
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, data, return_f_matrix=False):
        """
        Forward pass through GCN layers.
        
        Args:
            data: PyG Data object with x, edge_index, batch
            return_f_matrix: If True, returns dummy matrix for API compatibility
            
        Returns:
            x: [B, max_nodes*L, C] where L=num_layers, matching scattering output format
        """
        x = data.x
        edge_index = data.edge_index
        
        # Ensure edge_index is on same device as x
        if edge_index.device != x.device:
            edge_index = edge_index.to(x.device)
        
        N, C = x.shape
        
        # Store outputs from each GCN layer (analogous to scattering coefficients)
        layer_outputs = []
        h = x
        
        for i, (gcn, bn, mlp) in enumerate(zip(self.gcn_layers, self.batch_norms, self.layer_mlps)):
            # GCN convolution
            h = gcn(h, edge_index)
            h = bn(h)
            h = self.activation(h)
            # h = self.dropout(h)
            
            # Apply MLP to this layer's output (like scattering)
            layer_out = mlp(h)
            layer_outputs.append(layer_out)
        
        # Stack outputs: [N, L, C] matching scattering format
        x = torch.stack(layer_outputs, dim=1)  # [N, L, C]
        
        # To dense batch: [B, max_nodes, L, C]
        x_dense = to_dense_batch(x, data.batch)[0]
        
        # Flatten to [B, max_nodes*L, C] to match scattering output
        x = x_dense.reshape(data.num_graphs, -1, self.out_shape())
        
        if return_f_matrix:
            # Return dummy matrix for API compatibility with scattering
            dummy_matrix = torch.zeros((self.num_layers, 17), device=x.device)
            return x, dummy_matrix
        
        return x
    
    def out_shape(self):
        """Output feature dimension (matching scattering API)"""
        return self.num_layers * self.in_channels


class SimpleGCN_Layer(nn.Module):
    """
    Even simpler GCN variant - just stacks GCN convolutions and outputs final representation.
    This is a more minimal baseline that doesn't try to mimic scattering's multi-scale structure.
    
    Args:
        in_channels: Input feature dimension
        max_graph_size: Maximum number of nodes
        num_layers: Number of GCN layers
        output_dim: Output dimension per node
    """
    
    def __init__(self, in_channels, max_graph_size, num_layers=3, output_dim=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.max_graph_size = max_graph_size
        self.num_layers = num_layers
        self.output_dim = output_dim or (in_channels * num_layers)
        
        # Simple GCN stack
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else self.output_dim // num_layers
            out_dim = self.output_dim // num_layers
            
            self.gcn_layers.append(GCNConv(in_dim, out_dim))
            self.batch_norms.append(nn.BatchNorm1d(out_dim))
        
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
    
    def forward(self, data, return_f_matrix=False):
        """
        Simple forward pass - just propagate through GCN layers and concatenate.
        
        Returns:
            x: [B, max_nodes, output_dim]
        """
        x = data.x
        edge_index = data.edge_index
        
        if edge_index.device != x.device:
            edge_index = edge_index.to(x.device)
        
        layer_outputs = []
        h = x
        
        for gcn, bn in zip(self.gcn_layers, self.batch_norms):
            h = gcn(h, edge_index)
            h = bn(h)
            h = self.activation(h)
            # h = self.dropout(h)
            layer_outputs.append(h)
        
        # Concatenate all layer outputs: [N, output_dim]
        x = torch.cat(layer_outputs, dim=-1)
        
        # To dense batch: [B, max_nodes, output_dim]
        x_dense = to_dense_batch(x, data.batch)[0]
        
        # Reshape to [B, max_nodes, output_dim] - simplified format
        x = x_dense
        
        if return_f_matrix:
            dummy_matrix = torch.zeros((self.num_layers, 17), device=x.device)
            return x, dummy_matrix
        
        return x
    
    def out_shape(self):
        """Output feature dimension"""
        return self.output_dim

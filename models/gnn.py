"""
SimuStruct AI V3 — Graph Neural Network (GNN) for Stress Prediction
====================================================================
Model B: Graph Attention Network operating on the FEM mesh graph.
Each node's features include coordinates, material properties, and geometry.
Edge connectivity comes directly from the mesh triangulation.

This architecture naturally handles arbitrary mesh topologies and
spatially-varying stress concentrations near hole boundaries.

Requires: torch_geometric (pip install torch-geometric)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

# Conditional import for torch_geometric
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    # Placeholders for type hints
    Data = object
    Batch = object


class StressGNN(nn.Module):
    """
    Graph Attention Network for per-node stress field prediction.

    Architecture:
        - Linear encoder to hidden dimension
        - 3 GATConv layers with multi-head attention (4 heads each)
        - Decoder MLP for per-node output

    Input:
        data.x: (N, node_feat_dim) — per-node features
        data.edge_index: (2, E) — mesh edge connectivity

    Output:
        (N, 3) — [σ_vm, disp_x, disp_y] per node
    """

    def __init__(
        self,
        node_feat_dim: int = 15,
        hidden: int = 128,
        n_heads: int = 4,
        n_gat_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert HAS_TORCH_GEOMETRIC, (
            "torch_geometric is required for GNN model. "
            "Install: pip install torch-geometric"
        )

        self.node_feat_dim = node_feat_dim

        # Input encoder
        self.encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for _ in range(n_gat_layers):
            self.gat_layers.append(
                GATConv(hidden, hidden, heads=n_heads, concat=False, dropout=dropout)
            )
            self.gat_norms.append(nn.LayerNorm(hidden))

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),  # [sigma_vm, disp_x, disp_y]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass on mesh graph.

        Args:
            data: PyTorch Geometric Data object with .x and .edge_index

        Returns:
            (N, 3) per-node predictions
        """
        x, edge_index = data.x, data.edge_index

        # Encode
        x = self.encoder(x)

        # GAT message passing with residual connections
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            x_res = x
            x = gat(x, edge_index)
            x = norm(x)
            x = torch.relu(x + x_res)  # Residual
            x = self.dropout(x)

        # Decode per node
        return self.decoder(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def mesh_to_graph(
    coords: np.ndarray,
    triangles: np.ndarray,
    node_features: np.ndarray,
    targets: Optional[np.ndarray] = None,
) -> "Data":
    """
    Convert a triangular FEM mesh to a PyTorch Geometric graph.

    Args:
        coords: (N, 2) node coordinates
        triangles: (T, 3) triangle connectivity (node indices)
        node_features: (N, F) per-node feature vector
        targets: (N, 3) optional per-node targets [σ_vm, disp_x, disp_y]

    Returns:
        PyTorch Geometric Data object
    """
    assert HAS_TORCH_GEOMETRIC, "torch_geometric required"

    # Extract edges from triangles (unique, bidirectional)
    edges = set()
    for tri in triangles:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))

    edge_arr = np.array(list(edges), dtype=np.int64)
    # Make bidirectional
    edge_index = np.concatenate([edge_arr, edge_arr[:, ::-1]], axis=0).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        pos=torch.tensor(coords, dtype=torch.float32),
    )

    if targets is not None:
        data.y = torch.tensor(targets, dtype=torch.float32)

    return data


def build_node_features(
    coords: np.ndarray,
    material_feats: np.ndarray,
    geo_feats: np.ndarray,
    load_feats: np.ndarray,
) -> np.ndarray:
    """
    Assemble per-node feature vector for GNN input.

    Args:
        coords: (N, 2) — node X, Y
        material_feats: (4,) or (N, 4) — material properties (broadcast if scalar)
        geo_feats: (4,) or (N, 4) — geometry parameters
        load_feats: (5,) or (N, 5) — load parameters

    Returns:
        (N, 15) feature array
    """
    n_nodes = len(coords)

    def broadcast(arr, target_shape):
        arr = np.asarray(arr)
        if arr.ndim == 1 and arr.shape[0] < target_shape:
            return np.tile(arr, (n_nodes, 1))
        return arr

    mat = broadcast(material_feats, n_nodes)
    geo = broadcast(geo_feats, n_nodes)
    load = broadcast(load_feats, n_nodes)

    return np.concatenate([coords, mat, geo, load], axis=1)


def generate_simple_mesh(
    width: float, height: float, n_x: int = 20, n_y: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple rectangular triangular mesh for testing.

    Returns:
        coords: (N, 2) node coordinates
        triangles: (T, 3) triangle connectivity
    """
    x = np.linspace(0, width, n_x)
    y = np.linspace(0, height, n_y)
    X, Y = np.meshgrid(x, y)
    coords = np.column_stack([X.ravel(), Y.ravel()])

    triangles = []
    for i in range(n_y - 1):
        for j in range(n_x - 1):
            n0 = i * n_x + j
            n1 = n0 + 1
            n2 = n0 + n_x
            n3 = n2 + 1
            triangles.append([n0, n1, n2])
            triangles.append([n1, n3, n2])

    return coords, np.array(triangles)


if __name__ == "__main__":
    if not HAS_TORCH_GEOMETRIC:
        print("torch_geometric not installed. Skipping GNN test.")
        print("Install: pip install torch-geometric")
    else:
        # Quick test
        model = StressGNN(node_feat_dim=15)
        print(f"StressGNN — Parameters: {model.count_parameters():,}")

        # Create test mesh
        coords, triangles = generate_simple_mesh(1.0, 0.5, 20, 20)
        feats = np.random.randn(len(coords), 15).astype(np.float32)
        data = mesh_to_graph(coords, triangles, feats)

        print(f"Mesh: {len(coords)} nodes, {len(triangles)} triangles, {data.edge_index.shape[1]} edges")

        out = model(data)
        print(f"Output: {out.shape} → per-node [σ_vm, disp_x, disp_y]")

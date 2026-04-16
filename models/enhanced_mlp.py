"""
SimuStruct AI V3 — Enhanced MLP with Fourier Features
======================================================
Model A: Multi-Layer Perceptron with Fourier positional encoding for
coordinate inputs. The Fourier encoding dramatically improves stress
gradient capture near hole boundaries.

Architecture:
    Input: 45 features = 32 (Fourier-encoded coords) + 4 (geometry) + 4 (material) + 5 (load)
    Hidden: [128, 256, 256, 128] with LayerNorm + GELU
    Output: 3 channels [σ_vm, disp_x, disp_y]

Reference:
    Tancik et al., "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional


class FourierEncoding(nn.Module):
    """
    Maps low-dimensional coordinates to high-dimensional Fourier features.
    For (x, y) → [sin(2^0·π·x), cos(2^0·π·x), ..., sin(2^L·π·y), cos(2^L·π·y)]

    This allows the MLP to learn high-frequency stress gradients that would
    otherwise require extremely deep networks.
    """

    def __init__(self, n_freqs: int = 8, input_dim: int = 2):
        super().__init__()
        self.n_freqs = n_freqs
        self.input_dim = input_dim
        # Frequency bands: [2^0, 2^1, ..., 2^(n_freqs-1)]
        freqs = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)
        # Register as buffer (not a parameter, but moves with device)
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        """Output dimensionality: input_dim × 2 (sin+cos) × n_freqs"""
        return self.input_dim * 2 * self.n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) — raw coordinates [X, Y]
        Returns:
            (batch, input_dim * 2 * n_freqs) — Fourier features
        """
        # x: (N, 2), freqs: (n_freqs,) → x_scaled: (N, 2, n_freqs)
        x_scaled = x.unsqueeze(-1) * self.freqs * math.pi
        # Concatenate sin and cos → (N, 2, 2*n_freqs) → flatten → (N, 4*n_freqs)
        encoded = torch.cat([torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1)
        return encoded.flatten(start_dim=1)


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and GELU activation."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class EnhancedMLP(nn.Module):
    """
    Enhanced Multi-Layer Perceptron for stress field prediction.

    Features:
    - Fourier positional encoding for coordinates
    - LayerNorm for stable training
    - GELU activation
    - Optional residual connections
    - Configurable width and depth

    Input Groups:
        coords:         (N, 2)  → Fourier encoded to (N, 32)
        material_feats: (N, 4)  → [E_norm, nu, sigma_y_norm, rho_norm]
        geo_feats:      (N, 4)  → [hole_rx, hole_ry, hole_cx, hole_cy]
        load_feats:     (N, 5)  → [magnitude, angle, type_onehot(3)]

    Output: (N, 3) → [σ_vm, disp_x, disp_y]
    """

    def __init__(
        self,
        n_material_features: int = 4,
        n_geo_features: int = 4,
        n_load_features: int = 5,
        n_freqs: int = 8,
        hidden_dims: Optional[List[int]] = None,
        use_residual: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 128]

        self.fourier = FourierEncoding(n_freqs=n_freqs, input_dim=2)
        coord_dim = self.fourier.output_dim  # 2 * 2 * n_freqs = 32

        total_input = coord_dim + n_material_features + n_geo_features + n_load_features
        self.use_residual = use_residual

        # Build network layers
        layers = []
        in_dim = total_input
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            in_dim = h

        self.net = nn.Sequential(*layers)

        # Optional residual blocks
        if use_residual and len(hidden_dims) >= 2:
            self.res_block = ResidualBlock(hidden_dims[-1])
        else:
            self.res_block = nn.Identity()

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.GELU(),
            nn.Linear(64, 3),  # [sigma_vm, disp_x, disp_y]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        coords: torch.Tensor,
        material_feats: torch.Tensor,
        geo_feats: torch.Tensor,
        load_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            coords: (N, 2) raw X, Y coordinates
            material_feats: (N, 4) material properties
            geo_feats: (N, 4) geometry parameters
            load_feats: (N, 5) load parameters (optional)

        Returns:
            (N, 3) predictions [σ_vm, disp_x, disp_y]
        """
        # Fourier encode coordinates
        coord_enc = self.fourier(coords)  # (N, 32)

        # Concatenate all features
        parts = [coord_enc, material_feats, geo_feats]
        if load_feats is not None:
            parts.append(load_feats)
        x = torch.cat(parts, dim=-1)

        # Forward through MLP
        x = self.net(x)
        x = self.res_block(x)
        return self.head(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_enhanced_mlp(config: dict = None) -> EnhancedMLP:
    """Factory function to create EnhancedMLP from config dict."""
    if config is None:
        config = {}
    return EnhancedMLP(
        n_material_features=config.get("n_material_features", 4),
        n_geo_features=config.get("n_geo_features", 4),
        n_load_features=config.get("n_load_features", 5),
        n_freqs=config.get("n_freqs", 8),
        hidden_dims=config.get("hidden_dims", [128, 256, 256, 128]),
        use_residual=config.get("use_residual", True),
        dropout_p=config.get("dropout_p", 0.0),
    )


if __name__ == "__main__":
    # Quick test
    model = build_enhanced_mlp()
    print(f"EnhancedMLP — Parameters: {model.count_parameters():,}")

    # Test forward pass
    batch = 1024
    coords = torch.randn(batch, 2)
    mat = torch.randn(batch, 4)
    geo = torch.randn(batch, 4)
    load = torch.randn(batch, 5)

    out = model(coords, mat, geo, load)
    print(f"Input:  coords={coords.shape}, mat={mat.shape}, geo={geo.shape}, load={load.shape}")
    print(f"Output: {out.shape} → [σ_vm, disp_x, disp_y]")
    print(f"Sample output: {out[0].detach().numpy()}")

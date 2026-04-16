"""
SimuStruct AI V3 — Monte Carlo Dropout Uncertainty Quantification
==================================================================
Wraps the EnhancedMLP with permanent dropout layers to enable
uncertainty estimation via MC Dropout inference.

At inference time, we run N forward passes with dropout active,
collecting a distribution of predictions. The mean is the estimate
and the standard deviation quantifies epistemic uncertainty.

Reference:
    Gal & Ghahramani, "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning" (ICML 2016)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List

from models.enhanced_mlp import EnhancedMLP, FourierEncoding


class MCDropoutMLP(EnhancedMLP):
    """
    Enhanced MLP with permanent MC Dropout for uncertainty quantification.

    During inference, dropout remains active (model.train() mode) to sample
    from the approximate posterior distribution.
    """

    def __init__(
        self,
        n_material_features: int = 4,
        n_geo_features: int = 4,
        n_load_features: int = 5,
        n_freqs: int = 8,
        hidden_dims: Optional[List[int]] = None,
        dropout_p: float = 0.1,
    ):
        # Initialize parent with dropout
        super().__init__(
            n_material_features=n_material_features,
            n_geo_features=n_geo_features,
            n_load_features=n_load_features,
            n_freqs=n_freqs,
            hidden_dims=hidden_dims or [128, 256, 256, 128],
            use_residual=True,
            dropout_p=dropout_p,
        )
        self.mc_dropout_p = dropout_p

        # Add explicit dropout layers between each hidden layer
        self.mc_dropouts = nn.ModuleList([
            nn.Dropout(p=dropout_p) for _ in (hidden_dims or [128, 256, 256, 128])
        ])

    def forward(
        self,
        coords: torch.Tensor,
        material_feats: torch.Tensor,
        geo_feats: torch.Tensor,
        load_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass — dropout always active for MC sampling."""
        # Use parent forward
        return super().forward(coords, material_feats, geo_feats, load_feats)


def predict_with_uncertainty(
    model: MCDropoutMLP,
    coords: torch.Tensor,
    material_feats: torch.Tensor,
    geo_feats: torch.Tensor,
    load_feats: Optional[torch.Tensor] = None,
    n_samples: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run MC Dropout inference to get prediction mean and uncertainty.

    Args:
        model: MCDropoutMLP (must have dropout layers)
        coords, material_feats, geo_feats, load_feats: input tensors
        n_samples: number of forward passes (more = better uncertainty estimate)

    Returns:
        mean_pred: (N, 3) mean prediction
        std_pred: (N, 3) standard deviation (uncertainty)
        all_preds: (n_samples, N, 3) all sampled predictions
    """
    model.train()  # Keep dropout active during inference

    with torch.no_grad():
        predictions = []
        for _ in range(n_samples):
            pred = model(coords, material_feats, geo_feats, load_feats)
            predictions.append(pred)

    all_preds = torch.stack(predictions)       # (n_samples, N, 3)
    mean_pred = all_preds.mean(dim=0)           # (N, 3)
    std_pred = all_preds.std(dim=0)             # (N, 3)

    return mean_pred, std_pred, all_preds


def uncertainty_metrics(
    std_pred: torch.Tensor,
    target: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute uncertainty quality metrics.

    Args:
        std_pred: (N, 3) prediction uncertainty
        target: (N, 3) optional ground truth for calibration check

    Returns:
        Dictionary of uncertainty metrics
    """
    metrics = {
        "mean_uncertainty_stress": float(std_pred[:, 0].mean()),
        "max_uncertainty_stress": float(std_pred[:, 0].max()),
        "mean_uncertainty_disp_x": float(std_pred[:, 1].mean()),
        "mean_uncertainty_disp_y": float(std_pred[:, 2].mean()),
        "uncertainty_90th_pct": float(torch.quantile(std_pred[:, 0], 0.9)),
    }

    if target is not None:
        # Check if uncertainties are well-calibrated
        # (errors should fall within ±2σ for ~95% of points)
        from models.enhanced_mlp import EnhancedMLP
        # Placeholder — would need mean_pred too
        pass

    return metrics


def compute_confidence_intervals(
    mean_pred: torch.Tensor,
    std_pred: torch.Tensor,
    confidence: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute confidence intervals from MC Dropout predictions.

    Args:
        mean_pred: (N, 3) mean prediction
        std_pred: (N, 3) standard deviation
        confidence: confidence level (default 0.95 → ±1.96σ)

    Returns:
        lower_bound, upper_bound: (N, 3) each
    """
    from scipy.stats import norm
    z = norm.ppf((1 + confidence) / 2)

    lower = mean_pred - z * std_pred
    upper = mean_pred + z * std_pred

    return lower, upper


if __name__ == "__main__":
    model = MCDropoutMLP(dropout_p=0.1)
    print(f"MCDropoutMLP — Parameters: {model.count_parameters():,}")

    batch = 256
    coords = torch.randn(batch, 2)
    mat = torch.randn(batch, 4)
    geo = torch.randn(batch, 4)
    load = torch.randn(batch, 5)

    mean, std, all_preds = predict_with_uncertainty(
        model, coords, mat, geo, load, n_samples=30
    )
    print(f"Mean prediction shape: {mean.shape}")
    print(f"Uncertainty shape: {std.shape}")
    print(f"Mean stress uncertainty: {std[:, 0].mean():.4f}")
    print(f"Max stress uncertainty:  {std[:, 0].max():.4f}")

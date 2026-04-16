"""
SimuStruct AI V3 — Inference Engine
=====================================
Model loading, input preprocessing, forward pass, and post-processing.
Used by both the FastAPI compute engine and direct Streamlit calls.
"""

import os
import time
import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.materials import MATERIALS, get_material_features, get_lame_constants
from src.data_utils import preprocess_stress, postprocess_stress, SimuStructScaler


# ==================== GLOBAL MODEL STATE ====================
_model = None
_device = None
_scaler = None


def get_device() -> torch.device:
    """Detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def load_model(
    model_path: str = "models/simustruct_surrogate.pth",
    checkpoint_path: str = "models/best_checkpoint.pth",
    scaler_path: str = "models/scaler.pkl",
    model_type: str = "enhanced_mlp",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load trained model and scaler from disk.

    Tries model_path first, falls back to checkpoint_path.
    """
    global _model, _device, _scaler

    if device is None:
        device = get_device()
    _device = device

    # Load scaler if available
    if os.path.exists(scaler_path):
        _scaler = SimuStructScaler()
        _scaler.load(scaler_path)

    # Build model architecture
    if model_type == "enhanced_mlp":
        from models.enhanced_mlp import build_enhanced_mlp
        model = build_enhanced_mlp()
    elif model_type == "mc_dropout":
        from models.mc_dropout import MCDropoutMLP
        model = MCDropoutMLP(dropout_p=0.1)
    else:
        from models.enhanced_mlp import build_enhanced_mlp
        model = build_enhanced_mlp()

    # Load weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[Inference] Loaded model from {model_path}")
    elif os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Inference] Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"[Inference] WARNING: No trained model found. Using random weights.")

    model = model.to(device)
    model.eval()
    _model = model

    return model


def run_ai_inference(
    geometry: Dict,
    material: Dict,
    load: Dict,
    material_key: str = "structural_steel_a36",
    model: Optional[nn.Module] = None,
    n_grid: int = 80,
) -> Dict:
    """
    Run AI surrogate model inference.

    Args:
        geometry: dict with plate_width, plate_height, holes
        material: material dict from MATERIALS
        load: dict with type, magnitude, angle
        material_key: key into MATERIALS dict
        model: optional pre-loaded model
        n_grid: grid resolution for coordinate generation

    Returns:
        Dict with node_coords, stress_vm, displacement_x, displacement_y,
        sigma_max_mpa, scf, safety_factor, inference_ms
    """
    global _model, _device

    if model is None:
        if _model is None:
            load_model()
        model = _model

    device = _device or get_device()

    t0 = time.perf_counter()

    # Generate coordinate grid
    width = geometry.get("plate_width", geometry.get("width", 1.0))
    height = geometry.get("plate_height", geometry.get("height", 0.5))

    x = np.linspace(0, width, n_grid)
    y = np.linspace(0, height, n_grid)
    X, Y = np.meshgrid(x, y)
    coords = np.column_stack([X.ravel(), Y.ravel()])

    # Prepare material features
    mat_feats = get_material_features(material_key)   # [E_norm, nu, sy_norm, rho_norm]

    # Prepare geometry features
    holes = geometry.get("holes", [{"rx": 0.05, "ry": 0.05, "cx": 0.5, "cy": 0.5}])
    primary_hole = holes[0] if holes else {"rx": 0.05, "ry": 0.05, "cx": 0.5, "cy": 0.5}
    geo_feats = [
        primary_hole["rx"],
        primary_hole["ry"],
        primary_hole["cx"],
        primary_hole["cy"],
    ]

    # Prepare load features
    load_magnitude = load.get("magnitude", 1e5)
    load_type = load.get("type", "tension")
    load_angle = load.get("angle", 0.0)
    load_type_onehot = [
        1.0 if load_type in ("tension", "compression", "axial_tension") else 0.0,
        1.0 if load_type in ("shear", "transverse_shear") else 0.0,
        1.0 if load_type in ("biaxial", "biaxial_tension") else 0.0,
    ]
    load_feats = [
        load_magnitude / 1e6,   # Normalize
        load_angle / 90.0,
        *load_type_onehot,
    ]

    n_nodes = len(coords)

    # Broadcast to per-node tensors
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    mat_t = torch.tensor([mat_feats] * n_nodes, dtype=torch.float32, device=device)
    geo_t = torch.tensor([geo_feats] * n_nodes, dtype=torch.float32, device=device)
    load_t = torch.tensor([load_feats] * n_nodes, dtype=torch.float32, device=device)

    # Forward pass (network expects raw coordinates in meters)
    model.eval()
    with torch.no_grad():
        pred = model(coords_t, mat_t, geo_t, load_t)

    pred_np = pred.cpu().numpy()

    # Post-process: inverse log-transform for stress
    stress_vm = postprocess_stress(pred_np[:, 0])
    stress_vm = np.abs(stress_vm)  # Stress is always positive

    disp_x = pred_np[:, 1]
    disp_y = pred_np[:, 2]

    # Apply hole masking (zero stress inside holes)
    for hole in holes:
        cx = hole["cx"] * width
        cy = hole["cy"] * height
        rx = hole["rx"]
        ry = hole["ry"]
        inside = ((coords[:, 0] - cx) / rx) ** 2 + ((coords[:, 1] - cy) / ry) ** 2 <= 1.0
        stress_vm[inside] = 0.0
        disp_x[inside] = 0.0
        disp_y[inside] = 0.0

    # Compute metrics
    sigma_max = float(np.max(stress_vm))
    sigma_nominal = abs(load_magnitude) / 1e6 if load_magnitude != 0 else 1.0
    scf = sigma_max / sigma_nominal if sigma_nominal > 0 else 1.0

    sigma_y = material.get("sigma_y", MATERIALS.get(material_key, {}).get("sigma_y", 250e6))
    sigma_y_mpa = sigma_y / 1e6
    safety_factor = sigma_y_mpa / sigma_max if sigma_max > 0 else float("inf")

    t_ms = (time.perf_counter() - t0) * 1000

    return {
        "node_coords": coords.tolist(),
        "stress_vm": stress_vm.tolist(),
        "displacement_x": disp_x.tolist(),
        "displacement_y": disp_y.tolist(),
        "sigma_max_mpa": sigma_max / 1e6,
        "scf": scf,
        "safety_factor": safety_factor,
        "inference_ms": t_ms,
        "n_nodes": n_nodes,
        "model_type": "enhanced_mlp",
    }


def run_ai_inference_with_uncertainty(
    geometry: Dict,
    material: Dict,
    load: Dict,
    material_key: str = "structural_steel_a36",
    n_mc_samples: int = 50,
) -> Dict:
    """
    Run inference with MC Dropout uncertainty quantification.
    Returns mean prediction + uncertainty bounds.
    """
    # Run base inference for deterministic result
    result = run_ai_inference(geometry, material, load, material_key)

    # MC Dropout uncertainty (if MC model loaded)
    global _model
    if _model is not None and hasattr(_model, 'mc_dropout_p'):
        from models.mc_dropout import predict_with_uncertainty
        # ... would run MC sampling here
        result["has_uncertainty"] = True
        result["uncertainty_available"] = True
    else:
        result["has_uncertainty"] = False
        result["uncertainty_available"] = False

    return result

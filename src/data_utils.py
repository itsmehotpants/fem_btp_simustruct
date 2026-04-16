"""
SimuStruct AI V3 — Data Utilities
==================================
Latin Hypercube Sampling, normalization, log-transforms, HDF5 I/O.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ==================== STRESS TRANSFORMS ====================

def preprocess_stress(sigma: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Log-transform stress values to handle exponential concentrations near holes."""
    return np.log(np.abs(sigma) + eps)


def postprocess_stress(log_sigma: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Inverse log-transform to recover original stress scale."""
    return np.exp(log_sigma) - eps


# ==================== NORMALIZATION ====================

class SimuStructScaler:
    """
    Per-channel normalization scaler for multi-feature input arrays.
    Stores means and standard deviations for consistent train/inference scaling.
    """

    def __init__(self):
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, X: Dict[str, np.ndarray]) -> "SimuStructScaler":
        """Compute per-channel statistics from training data."""
        for key, arr in X.items():
            self.means_[key] = float(arr.mean())
            self.stds_[key] = float(arr.std()) + 1e-8
        self.is_fitted = True
        return self

    def transform(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Normalize each channel to zero mean, unit variance."""
        assert self.is_fitted, "Scaler must be fitted before transform"
        cols = []
        for key, arr in X.items():
            cols.append((arr - self.means_[key]) / self.stds_[key])
        return np.column_stack(cols)

    def fit_transform(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray, keys: List[str]) -> Dict[str, np.ndarray]:
        """Reverse normalization back to original scale."""
        assert X.shape[1] == len(keys), "Column count must match keys"
        result = {}
        for i, key in enumerate(keys):
            result[key] = X[:, i] * self.stds_[key] + self.means_[key]
        return result

    def save(self, filepath: str):
        """Persist scaler state to pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump({"means": self.means_, "stds": self.stds_}, f)

    def load(self, filepath: str) -> "SimuStructScaler":
        """Load scaler state from pickle file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.means_ = data["means"]
        self.stds_ = data["stds"]
        self.is_fitted = True
        return self


# ==================== LATIN HYPERCUBE SAMPLING ====================

def generate_lhs_samples(n_samples: int = 5000, seed: int = 42) -> np.ndarray:
    """
    Generate Latin Hypercube samples across the full geometry + load parameter space.

    Returns:
        (n_samples, 10) array with columns:
        [plate_W, plate_H, hole_rx, hole_ry, hole_cx, hole_cy,
         mesh_size, load_tension, load_shear, load_biaxial_frac]
    """
    if HAS_SCIPY:
        sampler = qmc.LatinHypercube(d=10, seed=seed)
        samples = sampler.random(n=n_samples)
        l_bounds = [0.5, 0.3, 0.02, 0.02, 0.2, 0.2, 0.01, 1e4, 1e4, 0.0]
        u_bounds = [2.0, 1.5, 0.15, 0.15, 0.8, 0.8, 0.04, 1e6, 5e5, 1.0]
        return qmc.scale(samples, l_bounds, u_bounds)
    else:
        # Fallback: uniform random sampling
        rng = np.random.default_rng(seed)
        l_bounds = np.array([0.5, 0.3, 0.02, 0.02, 0.2, 0.2, 0.01, 1e4, 1e4, 0.0])
        u_bounds = np.array([2.0, 1.5, 0.15, 0.15, 0.8, 0.8, 0.04, 1e6, 5e5, 1.0])
        return rng.uniform(l_bounds, u_bounds, size=(n_samples, 10))


# ==================== HDF5 I/O ====================

def save_simulation_hdf5(result: dict, filepath: str, group_name: Optional[str] = None):
    """Save a single simulation result to an HDF5 file with gzip compression."""
    assert HAS_H5PY, "h5py is required for HDF5 I/O. Install with: pip install h5py"
    if group_name is None:
        group_name = f"sim_{uuid4().hex[:8]}"

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    with h5py.File(filepath, "a") as f:
        grp = f.create_group(group_name)
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                grp.create_dataset(key, data=val, compression="gzip", compression_opts=4)
            elif isinstance(val, (int, float)):
                grp.attrs[key] = val
            else:
                grp.attrs[key] = str(val)


def load_simulation_hdf5(filepath: str, group_name: str) -> dict:
    """Load a single simulation result from an HDF5 file."""
    assert HAS_H5PY, "h5py required"
    result = {}
    with h5py.File(filepath, "r") as f:
        grp = f[group_name]
        for key in grp.keys():
            result[key] = grp[key][:]
        for key, val in grp.attrs.items():
            result[key] = val
    return result


def load_all_simulations_hdf5(filepath: str) -> List[dict]:
    """Load all simulation groups from an HDF5 file."""
    assert HAS_H5PY, "h5py required"
    results = []
    with h5py.File(filepath, "r") as f:
        for group_name in f.keys():
            results.append(load_simulation_hdf5(filepath, group_name))
    return results


# ==================== NPZ FALLBACK ====================

def save_simulation_npz(result: dict, filepath: str):
    """Save simulation result as compressed .npz (fallback if h5py unavailable)."""
    arrays = {}
    metadata = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            arrays[key] = val
        else:
            metadata[key] = val

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    np.savez_compressed(filepath, **arrays, **{f"_meta_{k}": np.array([str(v)]) for k, v in metadata.items()})


def load_simulation_npz(filepath: str) -> dict:
    """Load simulation result from .npz file."""
    data = np.load(filepath, allow_pickle=True)
    result = {}
    for key in data.files:
        if key.startswith("_meta_"):
            result[key[6:]] = str(data[key][0])
        else:
            result[key] = data[key]
    return result


# ==================== DATASET BUILDER ====================

def build_training_dataset(
    data_dir: str,
    log_transform: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all simulation files and assemble into training arrays.

    Returns:
        (X_coords, X_features, Y_stress, Y_disp)
        - X_coords: (N_total, 2) node coordinates
        - X_features: (N_total, n_features) material + geometry + load features
        - Y_stress: (N_total,) Von Mises stress
        - Y_disp: (N_total, 2) displacement (x, y)
    """
    all_coords = []
    all_features = []
    all_stress = []
    all_disp = []

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".h5") or fname.endswith(".hdf5"):
            sims = load_all_simulations_hdf5(os.path.join(data_dir, fname))
        elif fname.endswith(".npz"):
            sims = [load_simulation_npz(os.path.join(data_dir, fname))]
        else:
            continue

        for sim in sims:
            coords = sim.get("coords", sim.get("node_coords"))
            stress = sim.get("stress_vm")
            disp_x = sim.get("disp_x", sim.get("displacement_x"))
            disp_y = sim.get("disp_y", sim.get("displacement_y"))

            if coords is None or stress is None:
                continue

            n_nodes = len(coords)

            # Build per-node feature vector (broadcast scalar params)
            feat_cols = []
            for fkey in ["E_norm", "nu", "sigma_y_norm", "rho_norm",
                         "hole_rx", "hole_ry", "hole_cx", "hole_cy",
                         "load_mag", "load_angle", "load_type_0",
                         "load_type_1", "load_type_2"]:
                if fkey in sim:
                    val = sim[fkey]
                    if isinstance(val, np.ndarray) and len(val) == n_nodes:
                        feat_cols.append(val)
                    else:
                        feat_cols.append(np.full(n_nodes, float(val)))

            if feat_cols:
                features = np.column_stack(feat_cols)
            else:
                features = np.zeros((n_nodes, 1))

            all_coords.append(coords if isinstance(coords, np.ndarray) else np.array(coords))
            all_features.append(features)

            stress_arr = stress if isinstance(stress, np.ndarray) else np.array(stress)
            if log_transform:
                stress_arr = preprocess_stress(stress_arr)
            all_stress.append(stress_arr)

            dx = disp_x if isinstance(disp_x, np.ndarray) else np.array(disp_x if disp_x is not None else [0]*n_nodes)
            dy = disp_y if isinstance(disp_y, np.ndarray) else np.array(disp_y if disp_y is not None else [0]*n_nodes)
            all_disp.append(np.column_stack([dx, dy]))

    if not all_coords:
        raise ValueError(f"No valid simulation files found in {data_dir}")

    return (
        np.concatenate(all_coords, axis=0),
        np.concatenate(all_features, axis=0),
        np.concatenate(all_stress, axis=0),
        np.concatenate(all_disp, axis=0),
    )

#!/usr/bin/env python3
"""
SimuStruct AI V3 — Data Generation Pipeline
=============================================
Generates parametric FEM / analytical stress field data for training.
Uses Latin Hypercube Sampling to cover the full geometry × material × load space.

Usage:
    python 1_generate_data.py                    # Generate 5000 samples
    python 1_generate_data.py --n_samples 100    # Quick test run
    python 1_generate_data.py --resume           # Resume from last checkpoint
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.materials import MATERIALS, get_material_features
from src.data_utils import (
    generate_lhs_samples,
    save_simulation_npz,
    SimuStructScaler,
)
from src.fem_solver import analytical_stress_field


LOAD_TYPES = ["tension", "shear", "biaxial", "compression", "combined"]


def generate_dataset(
    n_samples: int = 5000,
    output_dir: str = "data/hdf5",
    resume: bool = False,
    seed: int = 42,
):
    """
    Generate the full parametric dataset.

    For each sample:
    1. Draw geometry + load from LHS
    2. Pick a random material
    3. Run analytical solver (or FEniCSx if available)
    4. Save result with all features
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Check for resume
    existing = set()
    if resume:
        for f in os.listdir(output_dir):
            if f.endswith(".npz"):
                existing.add(f)
        print(f"[RESUME] Found {len(existing)} existing samples, will skip them.")

    # Generate LHS samples
    print(f"\n{'='*60}")
    print(f"  SimuStruct AI V3 — Data Generation Pipeline")
    print(f"  Samples: {n_samples} | Materials: {len(MATERIALS)}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}\n")

    lhs_samples = generate_lhs_samples(n_samples=n_samples, seed=seed)
    material_keys = list(MATERIALS.keys())
    rng = np.random.default_rng(seed)

    # Track statistics
    stats = {
        "total": n_samples,
        "completed": 0,
        "failed": 0,
        "skipped": len(existing),
        "start_time": datetime.now().isoformat(),
        "materials_used": {},
    }

    t_start = time.time()

    for i in range(n_samples):
        sample_id = f"sim_{i:05d}"
        fname = f"{sample_id}.npz"

        if fname in existing:
            continue

        # Extract geometry from LHS
        s = lhs_samples[i]
        plate_width  = float(s[0])
        plate_height = float(s[1])
        hole_rx      = float(s[2])
        hole_ry      = float(s[3])
        hole_cx      = float(s[4])
        hole_cy      = float(s[5])
        mesh_size    = float(s[6])
        load_tension = float(s[7])
        load_shear   = float(s[8])
        load_biax    = float(s[9])

        # Random material
        mat_key = rng.choice(material_keys)

        # Random load type
        load_type = rng.choice(LOAD_TYPES)
        if load_type == "tension":
            load_mag = load_tension
        elif load_type == "shear":
            load_mag = load_shear
        elif load_type == "biaxial":
            load_mag = load_biax
        elif load_type == "compression":
            load_mag = load_tension
        else:  # combined
            load_mag = load_tension

        # Number of holes (1-3)
        n_holes = rng.integers(1, 4)
        holes = []
        for h_idx in range(n_holes):
            if h_idx == 0:
                holes.append({
                    "rx": hole_rx, "ry": hole_ry,
                    "cx": hole_cx, "cy": hole_cy
                })
            else:
                # Random additional holes
                holes.append({
                    "rx": rng.uniform(0.02, 0.12),
                    "ry": rng.uniform(0.02, 0.12),
                    "cx": rng.uniform(0.2, 0.8),
                    "cy": rng.uniform(0.2, 0.8),
                })

        try:
            # Run solver
            result = analytical_stress_field(
                width=plate_width,
                height=plate_height,
                holes=holes,
                material_key=mat_key,
                load_magnitude=load_mag,
                load_type=load_type,
                n_grid=80,
            )

            # Compute material features for AI input
            mat_feats = get_material_features(mat_key)
            load_angle = 0.0 if load_type in ("tension", "compression") else (
                90.0 if load_type == "shear" else 45.0
            )
            load_type_onehot = [
                1.0 if load_type in ("tension", "compression") else 0.0,
                1.0 if load_type == "shear" else 0.0,
                1.0 if load_type == "biaxial" else 0.0,
            ]

            # Augment result with features
            n_nodes = len(result["coords"])
            result["E_norm"] = np.full(n_nodes, mat_feats[0])
            result["nu"] = np.full(n_nodes, mat_feats[1])
            result["sigma_y_norm"] = np.full(n_nodes, mat_feats[2])
            result["rho_norm"] = np.full(n_nodes, mat_feats[3])
            result["hole_rx"] = np.full(n_nodes, hole_rx)
            result["hole_ry"] = np.full(n_nodes, hole_ry)
            result["hole_cx"] = np.full(n_nodes, hole_cx)
            result["hole_cy"] = np.full(n_nodes, hole_cy)
            result["load_mag"] = np.full(n_nodes, load_mag / 1e6)  # Normalize
            result["load_angle"] = np.full(n_nodes, load_angle / 90.0)
            result["load_type_0"] = np.full(n_nodes, load_type_onehot[0])
            result["load_type_1"] = np.full(n_nodes, load_type_onehot[1])
            result["load_type_2"] = np.full(n_nodes, load_type_onehot[2])

            # Save metadata as arrays for npz compatibility
            result["material_key"] = mat_key
            result["load_type"] = load_type
            result["plate_width"] = plate_width
            result["plate_height"] = plate_height
            result["n_holes"] = n_holes

            save_simulation_npz(result, os.path.join(output_dir, fname))

            stats["completed"] += 1
            mat_name = MATERIALS[mat_key]["display_name"]
            stats["materials_used"][mat_name] = stats["materials_used"].get(mat_name, 0) + 1

        except Exception as e:
            stats["failed"] += 1
            print(f"  [FAIL] {sample_id}: {e}")
            continue

        # Progress bar
        if (i + 1) % 50 == 0 or i == n_samples - 1:
            elapsed = time.time() - t_start
            rate = (stats["completed"]) / max(elapsed, 1)
            eta = (n_samples - i - 1) / max(rate, 0.01)
            pct = (i + 1) / n_samples * 100
            bar_len = 30
            filled = int(bar_len * (i + 1) / n_samples)
            bar = "#" * filled + "-" * (bar_len - filled)
            sigma_max = result.get("sigma_max_mpa", 0)
            print(
                f"\r  [{bar}] {pct:5.1f}% | "
                f"{stats['completed']}/{n_samples} | "
                f"sig_max={sigma_max:.1f} MPa | "
                f"{rate:.1f} sim/s | ETA: {eta:.0f}s",
                end="", flush=True,
            )

    print()  # Newline after progress bar

    # Save statistics
    stats["end_time"] = datetime.now().isoformat()
    stats["elapsed_seconds"] = time.time() - t_start
    stats_path = os.path.join("logs", "data_generation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Data generation complete!")
    print(f"  Completed: {stats['completed']} / {n_samples}")
    print(f"  Failed:    {stats['failed']}")
    print(f"  Time:      {stats['elapsed_seconds']:.1f}s")
    print(f"  Output:    {output_dir}/")
    print(f"  Stats:     {stats_path}")
    print(f"{'='*60}\n")

    # Fit and save scaler on generated data
    print("Fitting data scaler...")
    fit_scaler(output_dir)

    return stats


def fit_scaler(data_dir: str):
    """Fit the SimuStructScaler on all generated data and save."""
    from src.data_utils import SimuStructScaler
    import glob

    all_features = {
        "E_norm": [], "nu": [], "sigma_y_norm": [], "rho_norm": [],
        "hole_rx": [], "hole_ry": [], "hole_cx": [], "hole_cy": [],
        "load_mag": [], "load_angle": [],
        "load_type_0": [], "load_type_1": [], "load_type_2": [],
    }

    for fpath in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        data = np.load(fpath, allow_pickle=True)
        for key in all_features:
            if key in data:
                all_features[key].append(data[key])

    if not all_features["E_norm"]:
        print("  [WARN] No data files found for scaler fitting.")
        return

    # Concatenate
    for key in all_features:
        if all_features[key]:
            all_features[key] = np.concatenate(all_features[key])
        else:
            all_features[key] = np.array([0.0])

    scaler = SimuStructScaler()
    scaler.fit(all_features)

    os.makedirs("models", exist_ok=True)
    scaler.save("models/scaler.pkl")
    print(f"  Scaler saved to models/scaler.pkl")
    print(f"  Feature means: { {k: f'{v:.4f}' for k, v in scaler.means_.items()} }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimuStruct AI — Data Generation")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of simulations to generate (default: 500)")
    parser.add_argument("--output_dir", type=str, default="data/hdf5",
                        help="Output directory for simulation data")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing data files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        resume=args.resume,
        seed=args.seed,
    )

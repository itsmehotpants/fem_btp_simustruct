"""
SimuStruct AI V3 — Design Optimizer
=====================================
Uses differential_evolution to find optimal hole positions that
minimize peak stress, leveraging the AI surrogate for fast evaluation
(1000+ evaluations/second).
"""

import numpy as np
from typing import Dict, Optional, Callable, List, Tuple

from src.materials import MATERIALS
from src.inference import run_ai_inference


def optimize_hole_position(
    plate_width: float = 1.0,
    plate_height: float = 0.5,
    material_key: str = "structural_steel_a36",
    load_type: str = "tension",
    load_magnitude: float = 1e5,
    target_safety_factor: float = 2.0,
    n_holes: int = 1,
    fixed_rx: Optional[float] = None,
    fixed_ry: Optional[float] = None,
    max_iterations: int = 200,
    seed: int = 42,
    callback: Optional[Callable] = None,
) -> Dict:
    """
    Optimize hole position(s) to minimize maximum Von Mises stress.

    Uses scipy.optimize.differential_evolution with the AI surrogate
    as the objective function.

    Args:
        plate_width, plate_height: Plate dimensions (m)
        material_key: Material for the plate
        load_type, load_magnitude: Applied loading
        target_safety_factor: Desired safety factor
        n_holes: Number of holes to optimize (1-3)
        fixed_rx, fixed_ry: If provided, fix hole radii (only optimize position)
        max_iterations: Max DE iterations
        seed: Random seed
        callback: Optional callback(xk, convergence) for progress tracking

    Returns:
        Dict with optimal parameters, min stress, optimization history
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return {"error": "scipy is required for optimization. Install: pip install scipy"}

    mat = MATERIALS[material_key]
    history = {"iterations": [], "best_stress": [], "params": []}

    # Define bounds per hole: [cx, cy, rx, ry]
    bounds = []
    for _ in range(n_holes):
        bounds.append((0.15, 0.85))  # cx
        bounds.append((0.15, 0.85))  # cy
        if fixed_rx is None:
            bounds.append((0.02, 0.12))  # rx
        if fixed_ry is None:
            bounds.append((0.02, 0.12))  # ry

    eval_count = [0]

    def objective(x):
        """Objective: minimize peak Von Mises stress."""
        holes = []
        idx = 0
        for _ in range(n_holes):
            cx = x[idx]; idx += 1
            cy = x[idx]; idx += 1
            rx = x[idx] if fixed_rx is None else fixed_rx
            if fixed_rx is None:
                idx += 1
            ry = x[idx] if fixed_ry is None else fixed_ry
            if fixed_ry is None:
                idx += 1
            holes.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry})

        # Check for overlapping holes
        for i in range(len(holes)):
            for j in range(i + 1, len(holes)):
                dx = (holes[i]["cx"] - holes[j]["cx"]) * plate_width
                dy = (holes[i]["cy"] - holes[j]["cy"]) * plate_height
                dist = np.sqrt(dx**2 + dy**2)
                r_sum = max(holes[i]["rx"], holes[i]["ry"]) + max(holes[j]["rx"], holes[j]["ry"])
                if dist < r_sum * 1.5:
                    return 1e10  # Penalty for overlapping holes

        geometry = {
            "plate_width": plate_width,
            "plate_height": plate_height,
            "holes": holes,
        }
        load = {"type": load_type, "magnitude": load_magnitude, "angle": 0}

        try:
            result = run_ai_inference(geometry, mat, load, material_key, n_grid=50)
            sigma_max = result["sigma_max_mpa"]
        except Exception:
            sigma_max = 1e10

        eval_count[0] += 1
        if eval_count[0] % 50 == 0:
            history["iterations"].append(eval_count[0])
            history["best_stress"].append(sigma_max)

        return sigma_max

    # Run optimization
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=max_iterations,
        tol=0.01,
        workers=1,
        callback=callback,
        polish=True,
    )

    # Extract optimal parameters
    opt_holes = []
    idx = 0
    for _ in range(n_holes):
        cx = result.x[idx]; idx += 1
        cy = result.x[idx]; idx += 1
        rx = result.x[idx] if fixed_rx is None else fixed_rx
        if fixed_rx is None:
            idx += 1
        ry = result.x[idx] if fixed_ry is None else fixed_ry
        if fixed_ry is None:
            idx += 1
        opt_holes.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry})

    return {
        "optimal_holes": opt_holes,
        "min_sigma_max_mpa": float(result.fun),
        "safety_factor": float(mat["sigma_y"] / 1e6 / result.fun) if result.fun > 0 else float("inf"),
        "meets_target_sf": (mat["sigma_y"] / 1e6 / result.fun >= target_safety_factor) if result.fun > 0 else True,
        "target_safety_factor": target_safety_factor,
        "n_evaluations": eval_count[0],
        "optimization_success": result.success,
        "optimization_message": result.message,
        "history": history,
    }


def parameter_sweep(
    plate_width: float = 1.0,
    plate_height: float = 0.5,
    material_key: str = "structural_steel_a36",
    load_magnitude: float = 1e5,
    sweep_param: str = "hole_rx",
    sweep_range: Tuple[float, float] = (0.02, 0.15),
    n_points: int = 20,
) -> Dict[str, List[float]]:
    """
    Sweep a single parameter and record peak stress for contour/line plots.
    """
    mat = MATERIALS[material_key]
    values = np.linspace(sweep_range[0], sweep_range[1], n_points)
    stresses = []

    for val in values:
        holes = [{"cx": 0.5, "cy": 0.5, "rx": 0.05, "ry": 0.05}]
        if sweep_param == "hole_rx":
            holes[0]["rx"] = val
        elif sweep_param == "hole_ry":
            holes[0]["ry"] = val
        elif sweep_param == "hole_cx":
            holes[0]["cx"] = val
        elif sweep_param == "hole_cy":
            holes[0]["cy"] = val

        geometry = {
            "plate_width": plate_width,
            "plate_height": plate_height,
            "holes": holes,
        }
        load = {"type": "tension", "magnitude": load_magnitude, "angle": 0}

        try:
            result = run_ai_inference(geometry, mat, load, material_key, n_grid=50)
            stresses.append(result["sigma_max_mpa"])
        except Exception:
            stresses.append(0.0)

    return {
        "parameter": sweep_param,
        "values": values.tolist(),
        "sigma_max_mpa": stresses,
    }

"""
SimuStruct AI V3 — Fatigue Life Estimator
==========================================
Estimates fatigue life using the Basquin equation (S-N curve)
with Goodman mean-stress correction.

Theory:
    Basquin: N = (σ_f' / σ_a)^(1/b)
    Goodman: σ_a_corrected = σ_a / (1 - σ_mean / σ_y)

Reference:
    - Dowling, "Mechanical Behavior of Materials", Ch. 9-10
    - ASM Handbook Volume 19: Fatigue and Fracture
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from src.materials import MATERIALS


def estimate_fatigue_life(
    sigma_max_mpa: float,
    sigma_min_mpa: float = 0.0,
    material_key: str = "structural_steel_a36",
    correction: str = "goodman",
) -> Dict:
    """
    Estimate fatigue life (cycles to failure) from stress range.

    Args:
        sigma_max_mpa: Maximum stress in cycle (MPa)
        sigma_min_mpa: Minimum stress in cycle (MPa)
        material_key: Key into MATERIALS database
        correction: Mean stress correction method ('goodman', 'soderberg', 'gerber')

    Returns:
        Dict with cycles_to_failure, life_category, stress_ratio, etc.
    """
    mat = MATERIALS.get(material_key)
    if mat is None:
        raise ValueError(f"Unknown material: {material_key}")

    fatigue = mat.get("fatigue_params", {"sigma_f_prime": 1000, "b": -0.12})
    sigma_f_prime = fatigue["sigma_f_prime"]  # Fatigue strength coefficient (MPa)
    b = fatigue["b"]                          # Fatigue strength exponent

    # Stress components
    sigma_mean = (sigma_max_mpa + sigma_min_mpa) / 2
    sigma_amp = (sigma_max_mpa - sigma_min_mpa) / 2
    stress_ratio = sigma_min_mpa / sigma_max_mpa if sigma_max_mpa != 0 else 0

    # Material limits
    sigma_y = mat["sigma_y"] / 1e6  # Convert Pa to MPa
    sigma_uts = mat["UTS"] / 1e6

    # Mean stress correction
    if sigma_amp <= 0:
        return {
            "cycles_to_failure": float("inf"),
            "life_category": "Infinite",
            "sigma_mean_mpa": sigma_mean,
            "sigma_amplitude_mpa": sigma_amp,
            "stress_ratio": stress_ratio,
            "correction_method": correction,
        }

    if correction == "goodman":
        # Goodman: σ_a_corrected = σ_a / (1 - σ_mean / σ_uts)
        denominator = 1 - sigma_mean / sigma_uts if sigma_uts > 0 else 1
        if denominator <= 0:
            sigma_a_corrected = float("inf")
        else:
            sigma_a_corrected = sigma_amp / denominator
    elif correction == "soderberg":
        # Soderberg: σ_a_corrected = σ_a / (1 - σ_mean / σ_y)
        denominator = 1 - sigma_mean / sigma_y if sigma_y > 0 else 1
        if denominator <= 0:
            sigma_a_corrected = float("inf")
        else:
            sigma_a_corrected = sigma_amp / denominator
    elif correction == "gerber":
        # Gerber: σ_a_corrected = σ_a / (1 - (σ_mean / σ_uts)²)
        denominator = 1 - (sigma_mean / sigma_uts) ** 2 if sigma_uts > 0 else 1
        if denominator <= 0:
            sigma_a_corrected = float("inf")
        else:
            sigma_a_corrected = sigma_amp / denominator
    else:
        sigma_a_corrected = sigma_amp

    # Basquin equation: N = (σ_f' / σ_a_corrected)^(1/b)
    if sigma_a_corrected <= 0 or sigma_a_corrected == float("inf"):
        N_cycles = 0.0
    else:
        try:
            N_cycles = (sigma_f_prime / sigma_a_corrected) ** (1 / b)
            N_cycles = max(0, N_cycles)
        except (OverflowError, ZeroDivisionError):
            N_cycles = 0.0

    # Classify life regime
    if N_cycles > 1e7:
        life_category = "Infinite Life"
    elif N_cycles > 1e4:
        life_category = "High-Cycle Fatigue (HCF)"
    elif N_cycles > 1e2:
        life_category = "Low-Cycle Fatigue (LCF)"
    else:
        life_category = "Static Failure Risk"

    return {
        "cycles_to_failure": float(N_cycles),
        "life_category": life_category,
        "sigma_mean_mpa": sigma_mean,
        "sigma_amplitude_mpa": sigma_amp,
        "sigma_corrected_mpa": sigma_a_corrected if sigma_a_corrected != float("inf") else None,
        "stress_ratio": stress_ratio,
        "correction_method": correction,
        "fatigue_strength_coeff": sigma_f_prime,
        "fatigue_exponent": b,
    }


def generate_sn_curve(
    material_key: str = "structural_steel_a36",
    n_points: int = 100,
    sigma_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, List[float]]:
    """
    Generate S-N (Wöhler) curve data for plotting.

    Returns:
        Dict with 'N' (cycles) and 'S' (stress amplitude) arrays
    """
    mat = MATERIALS.get(material_key)
    if mat is None:
        raise ValueError(f"Unknown material: {material_key}")

    fatigue = mat.get("fatigue_params", {"sigma_f_prime": 1000, "b": -0.12})
    sigma_f_prime = fatigue["sigma_f_prime"]
    b = fatigue["b"]

    if sigma_range is None:
        sigma_max = sigma_f_prime
        sigma_min = mat["sigma_y"] / 1e6 * 0.1  # 10% of yield
    else:
        sigma_min, sigma_max = sigma_range

    # Generate stress amplitudes
    S = np.linspace(sigma_min, sigma_max, n_points)

    # Compute cycles for each stress level
    N = []
    for s in S:
        if s > 0:
            n = (sigma_f_prime / s) ** (1 / b)
            N.append(max(1, n))
        else:
            N.append(1e10)

    return {
        "N": [float(n) for n in N],
        "S": [float(s) for s in S],
        "material": mat["display_name"],
        "endurance_limit": float(mat["sigma_y"] / 1e6 * 0.5),  # Approximate
    }


def miner_cumulative_damage(
    stress_ranges_mpa: List[float],
    cycle_counts: List[int],
    material_key: str = "structural_steel_a36",
) -> Dict:
    """
    Palmgren-Miner cumulative damage rule: D = Σ(n_i / N_i)
    Failure when D ≥ 1.0

    Args:
        stress_ranges_mpa: List of stress ranges for each loading block
        cycle_counts: Number of cycles at each stress range

    Returns:
        Dict with cumulative damage, remaining life, and per-block breakdown
    """
    blocks = []
    total_damage = 0.0

    for sigma_range, n_cycles in zip(stress_ranges_mpa, cycle_counts):
        result = estimate_fatigue_life(sigma_range, 0.0, material_key)
        N_i = result["cycles_to_failure"]
        if N_i > 0:
            damage_i = n_cycles / N_i
        else:
            damage_i = float("inf")

        total_damage += damage_i
        blocks.append({
            "stress_range_mpa": sigma_range,
            "applied_cycles": n_cycles,
            "allowable_cycles": N_i,
            "damage_fraction": damage_i,
        })

    remaining_life_fraction = max(0, 1.0 - total_damage)

    return {
        "cumulative_damage": total_damage,
        "remaining_life_fraction": remaining_life_fraction,
        "is_failed": total_damage >= 1.0,
        "blocks": blocks,
    }

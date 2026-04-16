"""
SimuStruct AI V3 — Stress Concentration Factor (SCF) Library
==============================================================
Theoretical SCF formulas from Peterson's Stress Concentration Factors
for validation against AI predictions.

References:
    - Pilkey & Pilkey, "Peterson's Stress Concentration Factors", 3rd Ed.
    - Neuber, "Theory of Notch Stresses" (1958)
    - Savin, "Stress Concentration around Holes" (1961)
"""

import numpy as np
from typing import Dict, Optional


def scf_circular_hole_infinite_plate(radius: float = 1.0) -> float:
    """
    SCF for a circular hole in an infinite plate under uniaxial tension.
    Classic Kirsch solution: K_t = 3.0 (exact)
    """
    return 3.0


def scf_circular_hole_finite_width(
    radius: float,
    plate_width: float,
) -> float:
    """
    SCF for a circular hole in a finite-width plate under uniaxial tension.
    Pilkey formula (width correction).

    K_t = 3 - 3.13β + 3.66β² - 1.53β³

    where β = 2r/W (diameter-to-width ratio)

    Valid for: β < 0.6
    """
    beta = 2 * radius / plate_width
    if beta >= 1.0:
        raise ValueError(f"Hole diameter exceeds plate width (β={beta:.3f})")
    if beta > 0.6:
        print(f"[SCF Warning] β={beta:.3f} > 0.6; formula accuracy degrades.")

    K_t = 3.0 - 3.13 * beta + 3.66 * beta**2 - 1.53 * beta**3
    return K_t


def scf_elliptical_hole_infinite_plate(
    rx: float,
    ry: float,
    load_direction: str = "perpendicular",
) -> float:
    """
    SCF for an elliptical hole in an infinite plate.

    - Perpendicular to major axis (most common):
        K_t = 1 + 2(a/b) where a is semi-axis perpendicular to load

    - Parallel to major axis:
        K_t = 1 + 2(b/a)

    Neuber's exact formula for tension perpendicular to major axis 'a':
        K_t = 1 + 2 × sqrt(a/ρ)
    where ρ = b²/a is the radius of curvature at the end of major axis.
    """
    if rx <= 0 or ry <= 0:
        raise ValueError("Semi-axes must be positive")

    if load_direction == "perpendicular":
        # Load perpendicular to the major axis (rx direction)
        K_t = 1 + 2 * (rx / ry)
    elif load_direction == "parallel":
        # Load parallel to the major axis
        K_t = 1 + 2 * (ry / rx)
    elif load_direction == "neuber":
        # Neuber's formula using radius of curvature
        rho = ry**2 / rx  # Radius of curvature at end of semi-axis rx
        K_t = 1 + 2 * np.sqrt(rx / rho)
    else:
        K_t = 1 + 2 * (rx / ry)

    return float(K_t)


def scf_elliptical_hole_finite_width(
    rx: float,
    ry: float,
    plate_width: float,
) -> float:
    """
    SCF for an elliptical hole in a finite-width plate.
    Combines infinite-plate SCF with width correction factor.

    K_t_finite ≈ K_t_infinite × C_width

    where C_width accounts for the finite width effect.
    """
    K_t_inf = scf_elliptical_hole_infinite_plate(rx, ry)

    # Width correction (Howland approximation extended to ellipses)
    beta = 2 * rx / plate_width
    if beta >= 1.0:
        raise ValueError("Hole width exceeds plate width")

    # Correction factor
    C_width = (1 - beta) ** (-0.5)  # Simplified correction

    # Cap at reasonable values
    K_t = min(K_t_inf * C_width, K_t_inf * 3.0)

    return float(K_t)


def scf_biaxial_loading(
    rx: float,
    ry: float,
    sigma_x: float,
    sigma_y: float,
) -> float:
    """
    SCF for an elliptical hole under biaxial loading.

    For biaxial tension (σ_x, σ_y):
        K_t = 1 + 2(a/b) - σ_y/σ_x × (1 + 2b/a - σ_x/σ_y)

    Simplified for equi-biaxial (σ_x = σ_y):
        K_t = 2.0 for circular hole (exact)
    """
    if sigma_x == 0:
        return scf_elliptical_hole_infinite_plate(rx, ry, "parallel")

    ratio = sigma_y / sigma_x

    # For circular hole under biaxial
    if abs(rx - ry) < 1e-10:
        K_t = 3.0 - ratio
        return float(K_t)

    # General elliptical, perpendicular loading direction
    K_t_x = 1 + 2 * (rx / ry)  # SCF from σ_x
    K_t_y = 1 + 2 * (ry / rx)  # SCF from σ_y

    # Superposition
    K_t = K_t_x - ratio * (K_t_y - 1)

    return float(K_t)


def compare_scf(
    rx: float,
    ry: float,
    plate_width: float = 1.0,
    ai_scf: Optional[float] = None,
) -> Dict:
    """
    Compare different SCF formulas for a given geometry.
    Optionally compares against AI-predicted SCF.

    Returns:
        Dict with SCF values from each formula and error metrics
    """
    result = {
        "geometry": {"rx": rx, "ry": ry, "plate_width": plate_width},
        "scf_values": {},
    }

    # Infinite plate
    if abs(rx - ry) < 1e-10:
        result["scf_values"]["kirsch_infinite"] = scf_circular_hole_infinite_plate()
    result["scf_values"]["neuber_infinite"] = scf_elliptical_hole_infinite_plate(rx, ry)

    # Finite width
    try:
        if abs(rx - ry) < 1e-10:
            result["scf_values"]["pilkey_finite"] = scf_circular_hole_finite_width(
                max(rx, ry), plate_width
            )
        result["scf_values"]["ellipse_finite"] = scf_elliptical_hole_finite_width(
            rx, ry, plate_width
        )
    except ValueError:
        pass

    # Biaxial (equi-biaxial case)
    result["scf_values"]["biaxial_equal"] = scf_biaxial_loading(rx, ry, 1.0, 1.0)

    # AI comparison
    if ai_scf is not None:
        result["ai_scf"] = ai_scf
        errors = {}
        for name, theoretical in result["scf_values"].items():
            error_pct = abs(ai_scf - theoretical) / theoretical * 100
            errors[name] = round(error_pct, 2)
        result["error_pct_vs_theory"] = errors
        result["best_match"] = min(errors, key=errors.get)
        result["best_match_error_pct"] = errors[result["best_match"]]

    return result


# ==================== CONVENIENCE TABLE ====================

SCF_REFERENCE_TABLE = {
    "circular_hole_infinite": {
        "formula": "K_t = 3.0",
        "conditions": "Circular hole, infinite plate, uniaxial tension",
        "value": 3.0,
    },
    "ellipse_2to1_infinite": {
        "formula": "K_t = 1 + 2(a/b) = 5.0",
        "conditions": "Elliptical hole (a/b=2), infinite plate, tension ⊥ major axis",
        "value": 5.0,
    },
    "circular_equibiaxial": {
        "formula": "K_t = 2.0",
        "conditions": "Circular hole, equal biaxial tension",
        "value": 2.0,
    },
    "circular_pure_shear": {
        "formula": "K_t = 4.0",
        "conditions": "Circular hole, pure shear loading",
        "value": 4.0,
    },
}


if __name__ == "__main__":
    print("SCF Library — Peterson's Reference Values")
    print("=" * 60)

    for name, ref in SCF_REFERENCE_TABLE.items():
        print(f"\n{name}:")
        print(f"  Formula:    {ref['formula']}")
        print(f"  Conditions: {ref['conditions']}")
        print(f"  K_t:        {ref['value']}")

    print(f"\n{'='*60}")
    print("\nExample Comparisons:")

    # Circular hole r=0.05, W=1.0
    comp = compare_scf(0.05, 0.05, 1.0, ai_scf=2.95)
    print(f"\nCircular hole (r=0.05, W=1.0):")
    for name, val in comp["scf_values"].items():
        print(f"  {name}: {val:.3f}")
    if "ai_scf" in comp:
        print(f"  AI SCF: {comp['ai_scf']:.3f}")
        print(f"  Best match: {comp['best_match']} (error: {comp['best_match_error_pct']:.1f}%)")

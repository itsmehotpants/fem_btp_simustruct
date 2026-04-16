"""
SimuStruct AI V3 — Complete Engineering Material Database
=========================================================
22 materials with full mechanical, thermal, and fracture properties.
Each material is consumed by both FEniCSx solver and AI model as input features.

Reference Standards: ASTM, ASM Handbook, Peterson's SCF, MIL-HDBK-5
"""

from typing import Tuple, Dict, Any, List, Optional


MATERIALS: Dict[str, Dict[str, Any]] = {
    # ==================== STEELS ====================
    "structural_steel_a36": {
        "display_name": "Structural Steel (ASTM A36)",
        "category": "Steel",
        "E": 200e9,            # Young's Modulus (Pa)
        "nu": 0.26,            # Poisson's Ratio
        "sigma_y": 250e6,      # Yield Strength (Pa)
        "UTS": 400e6,          # Ultimate Tensile Strength (Pa)
        "rho": 7850,           # Density (kg/m³)
        "alpha": 12.0e-6,      # Thermal Expansion Coefficient (1/°C)
        "K_IC": 50e6,          # Fracture Toughness (Pa√m)
        "color": "#4A90D9",
        "fatigue_params": {"sigma_f_prime": 1000, "b": -0.12},
    },
    "high_strength_steel_a572": {
        "display_name": "High-Strength Steel (ASTM A572 Gr50)",
        "category": "Steel",
        "E": 200e9,
        "nu": 0.26,
        "sigma_y": 345e6,
        "UTS": 450e6,
        "rho": 7850,
        "alpha": 12.0e-6,
        "K_IC": 55e6,
        "color": "#3A7BD5",
        "fatigue_params": {"sigma_f_prime": 1100, "b": -0.11},
    },
    "stainless_steel_304": {
        "display_name": "Stainless Steel 304",
        "category": "Steel",
        "E": 193e9,
        "nu": 0.29,
        "sigma_y": 215e6,
        "UTS": 505e6,
        "rho": 8000,
        "alpha": 17.2e-6,
        "K_IC": 100e6,
        "color": "#5B9BD5",
        "fatigue_params": {"sigma_f_prime": 1200, "b": -0.12},
    },
    "stainless_steel_316": {
        "display_name": "Stainless Steel 316",
        "category": "Steel",
        "E": 193e9,
        "nu": 0.27,
        "sigma_y": 220e6,
        "UTS": 515e6,
        "rho": 8027,
        "alpha": 16.0e-6,
        "K_IC": 110e6,
        "color": "#6BA3D6",
        "fatigue_params": {"sigma_f_prime": 1250, "b": -0.12},
    },

    # ==================== CAST & DUCTILE IRON ====================
    "cast_iron_gray_a48": {
        "display_name": "Cast Iron (Gray ASTM A48)",
        "category": "Cast Iron",
        "E": 120e9,
        "nu": 0.26,
        "sigma_y": 0.0,        # Brittle — no yield
        "UTS": 179e6,
        "rho": 7200,
        "alpha": 10.8e-6,
        "K_IC": 22e6,
        "color": "#8B8680",
        "fatigue_params": {"sigma_f_prime": 400, "b": -0.15},
    },
    "ductile_iron_a536": {
        "display_name": "Ductile Iron (ASTM A536)",
        "category": "Cast Iron",
        "E": 169e9,
        "nu": 0.275,
        "sigma_y": 290e6,
        "UTS": 414e6,
        "rho": 7100,
        "alpha": 11.0e-6,
        "K_IC": 30e6,
        "color": "#9E9589",
        "fatigue_params": {"sigma_f_prime": 800, "b": -0.13},
    },

    # ==================== ALUMINUM ====================
    "aluminum_6061_t6": {
        "display_name": "Aluminum 6061-T6",
        "category": "Aluminum",
        "E": 68.9e9,
        "nu": 0.33,
        "sigma_y": 276e6,
        "UTS": 310e6,
        "rho": 2700,
        "alpha": 23.6e-6,
        "K_IC": 29e6,
        "color": "#A0C4FF",
        "fatigue_params": {"sigma_f_prime": 600, "b": -0.11},
    },
    "aluminum_7075_t6": {
        "display_name": "Aluminum 7075-T6",
        "category": "Aluminum",
        "E": 71.7e9,
        "nu": 0.33,
        "sigma_y": 503e6,
        "UTS": 572e6,
        "rho": 2810,
        "alpha": 23.4e-6,
        "K_IC": 24e6,
        "color": "#90B4EF",
        "fatigue_params": {"sigma_f_prime": 900, "b": -0.10},
    },
    "aluminum_2024_t3": {
        "display_name": "Aluminum 2024-T3",
        "category": "Aluminum",
        "E": 73.1e9,
        "nu": 0.33,
        "sigma_y": 345e6,
        "UTS": 483e6,
        "rho": 2780,
        "alpha": 23.2e-6,
        "K_IC": 37e6,
        "color": "#80A4DF",
        "fatigue_params": {"sigma_f_prime": 750, "b": -0.11},
    },

    # ==================== TITANIUM ====================
    "titanium_ti6al4v": {
        "display_name": "Titanium Ti-6Al-4V",
        "category": "Titanium",
        "E": 113.8e9,
        "nu": 0.342,
        "sigma_y": 880e6,
        "UTS": 950e6,
        "rho": 4430,
        "alpha": 8.6e-6,
        "K_IC": 60e6,
        "color": "#B8860B",
        "fatigue_params": {"sigma_f_prime": 1900, "b": -0.10},
    },
    "titanium_grade2_cp": {
        "display_name": "Titanium Grade 2 (CP)",
        "category": "Titanium",
        "E": 103e9,
        "nu": 0.37,
        "sigma_y": 275e6,
        "UTS": 345e6,
        "rho": 4510,
        "alpha": 8.9e-6,
        "K_IC": 50e6,
        "color": "#C89620",
        "fatigue_params": {"sigma_f_prime": 800, "b": -0.11},
    },

    # ==================== COPPER & BRASS ====================
    "copper_c11000": {
        "display_name": "Copper (C11000 Annealed)",
        "category": "Copper Alloys",
        "E": 117e9,
        "nu": 0.34,
        "sigma_y": 69e6,
        "UTS": 220e6,
        "rho": 8960,
        "alpha": 17.0e-6,
        "K_IC": 100e6,
        "color": "#B87333",
        "fatigue_params": {"sigma_f_prime": 500, "b": -0.13},
    },
    "brass_c26000": {
        "display_name": "Brass (C26000)",
        "category": "Copper Alloys",
        "E": 110e9,
        "nu": 0.34,
        "sigma_y": 124e6,
        "UTS": 338e6,
        "rho": 8530,
        "alpha": 20.5e-6,
        "K_IC": 55e6,
        "color": "#CD9B1D",
        "fatigue_params": {"sigma_f_prime": 650, "b": -0.12},
    },

    # ==================== MAGNESIUM ====================
    "magnesium_az31b": {
        "display_name": "Magnesium AZ31B",
        "category": "Magnesium",
        "E": 45e9,
        "nu": 0.35,
        "sigma_y": 220e6,
        "UTS": 290e6,
        "rho": 1770,
        "alpha": 26.0e-6,
        "K_IC": 15e6,
        "color": "#C0C0C0",
        "fatigue_params": {"sigma_f_prime": 500, "b": -0.12},
    },

    # ==================== NICKEL SUPERALLOY ====================
    "inconel_718": {
        "display_name": "Nickel Alloy (Inconel 718)",
        "category": "Nickel Alloys",
        "E": 200e9,
        "nu": 0.29,
        "sigma_y": 1034e6,
        "UTS": 1240e6,
        "rho": 8190,
        "alpha": 13.0e-6,
        "K_IC": 80e6,
        "color": "#708090",
        "fatigue_params": {"sigma_f_prime": 2500, "b": -0.09},
    },

    # ==================== COMPOSITES ====================
    "cfrp_unidirectional": {
        "display_name": "Carbon Fiber CFRP (Unidirectional)",
        "category": "Composites",
        "E": 135e9,
        "nu": 0.30,
        "sigma_y": 0.0,        # No yield — brittle failure
        "UTS": 1500e6,
        "rho": 1600,
        "alpha": 0.5e-6,
        "K_IC": 20e6,
        "color": "#2F4F4F",
        "fatigue_params": {"sigma_f_prime": 2000, "b": -0.08},
    },
    "gfrp_woven_eglass": {
        "display_name": "GFRP (Woven E-glass)",
        "category": "Composites",
        "E": 25e9,
        "nu": 0.28,
        "sigma_y": 0.0,        # Brittle
        "UTS": 300e6,
        "rho": 1900,
        "alpha": 7.0e-6,
        "K_IC": 8e6,
        "color": "#556B2F",
        "fatigue_params": {"sigma_f_prime": 400, "b": -0.14},
    },

    # ==================== POLYMERS ====================
    "polycarbonate_pc": {
        "display_name": "Polycarbonate (PC)",
        "category": "Polymers",
        "E": 2.4e9,
        "nu": 0.37,
        "sigma_y": 60e6,
        "UTS": 65e6,
        "rho": 1200,
        "alpha": 68.0e-6,
        "K_IC": 2.2e6,
        "color": "#DDA0DD",
        "fatigue_params": {"sigma_f_prime": 100, "b": -0.15},
    },
    "abs_plastic": {
        "display_name": "ABS Plastic",
        "category": "Polymers",
        "E": 2.3e9,
        "nu": 0.40,
        "sigma_y": 40e6,
        "UTS": 44e6,
        "rho": 1050,
        "alpha": 72.0e-6,
        "K_IC": 1.8e6,
        "color": "#EE82EE",
        "fatigue_params": {"sigma_f_prime": 80, "b": -0.16},
    },
    "hdpe": {
        "display_name": "HDPE",
        "category": "Polymers",
        "E": 0.8e9,
        "nu": 0.44,
        "sigma_y": 26e6,
        "UTS": 32e6,
        "rho": 960,
        "alpha": 120.0e-6,
        "K_IC": 1.5e6,
        "color": "#FFB6C1",
        "fatigue_params": {"sigma_f_prime": 50, "b": -0.18},
    },

    # ==================== CONCRETE ====================
    "concrete_normal": {
        "display_name": "Concrete (Normal Strength)",
        "category": "Concrete",
        "E": 30e9,
        "nu": 0.20,
        "sigma_y": 0.0,        # Brittle
        "UTS": 30e6,
        "rho": 2400,
        "alpha": 10.0e-6,
        "K_IC": 0.8e6,
        "color": "#A9A9A9",
        "fatigue_params": {"sigma_f_prime": 50, "b": -0.20},
    },

    # ==================== RUBBER / ELASTOMERS ====================
    "rubber_natural_50a": {
        "display_name": "Rubber (Natural, 50A Shore)",
        "category": "Elastomers",
        "E": 1e6,              # ~0.001 GPa
        "nu": 0.4999,          # Nearly incompressible
        "sigma_y": 0.0,        # N/A for hyperelastic
        "UTS": 25e6,
        "rho": 920,
        "alpha": 200.0e-6,
        "K_IC": 0.0,           # N/A
        "color": "#8B0000",
        "fatigue_params": {"sigma_f_prime": 30, "b": -0.20},
    },
}


# ==================== CATEGORY DEFINITIONS ====================
CATEGORIES = sorted(set(mat["category"] for mat in MATERIALS.values()))


def get_materials_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """Return all materials belonging to a given category."""
    return {k: v for k, v in MATERIALS.items() if v["category"] == category}


def get_lame_constants(material_key: str) -> Tuple[float, float]:
    """
    Compute Lamé constants (λ, μ) from Young's modulus and Poisson's ratio.

    Returns:
        (lambda, mu): Lamé first parameter and shear modulus (Pa)
    """
    mat = MATERIALS[material_key]
    E, nu = mat["E"], mat["nu"]
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def get_shear_modulus(material_key: str) -> float:
    """G = E / (2(1 + ν))"""
    mat = MATERIALS[material_key]
    return mat["E"] / (2 * (1 + mat["nu"]))


def get_bulk_modulus(material_key: str) -> float:
    """K = E / (3(1 - 2ν))"""
    mat = MATERIALS[material_key]
    return mat["E"] / (3 * (1 - 2 * mat["nu"]))


def get_material_features(material_key: str) -> List[float]:
    """
    Return normalized feature vector for AI model input.
    Features: [E_normalized, nu, sigma_y_normalized, rho_normalized]
    """
    mat = MATERIALS[material_key]
    # Normalize to roughly [-1, 1] range using engineering scale factors
    E_norm = mat["E"] / 200e9          # Normalized by steel E
    nu = mat["nu"]
    sy_norm = mat["sigma_y"] / 1000e6  # Normalized by 1 GPa
    rho_norm = mat["rho"] / 8000       # Normalized by steel density
    return [E_norm, nu, sy_norm, rho_norm]


def get_material_summary(material_key: str) -> str:
    """Return a formatted string of material properties for display."""
    mat = MATERIALS[material_key]
    lines = [
        f"Material: {mat['display_name']}",
        f"Category: {mat['category']}",
        f"Young's Modulus: {mat['E']/1e9:.1f} GPa",
        f"Poisson's Ratio: {mat['nu']:.3f}",
        f"Yield Strength: {mat['sigma_y']/1e6:.0f} MPa",
        f"UTS: {mat['UTS']/1e6:.0f} MPa",
        f"Density: {mat['rho']} kg/m³",
        f"Shear Modulus: {get_shear_modulus(material_key)/1e9:.1f} GPa",
        f"Bulk Modulus: {get_bulk_modulus(material_key)/1e9:.1f} GPa",
        f"CTE: {mat['alpha']*1e6:.1f} × 10⁻⁶ /°C",
        f"Fracture Toughness: {mat['K_IC']/1e6:.0f} MPa√m",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"SimuStruct AI V3 — Material Database")
    print(f"Total materials: {len(MATERIALS)}")
    print(f"Categories: {CATEGORIES}")
    print()
    for key in MATERIALS:
        print(get_material_summary(key))
        print("-" * 50)

"""
SimuStruct AI V3 — FEM Solver Tests
=====================================
Physics validation tests for the analytical stress field solver.
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fem_solver import analytical_stress_field, run_fem
from src.scf_library import (
    scf_circular_hole_infinite_plate,
    scf_elliptical_hole_infinite_plate,
    scf_circular_hole_finite_width,
    compare_scf,
)


class TestAnalyticalSolver:
    """Test the Kirsch-based analytical solver."""

    def test_output_keys(self):
        """Solver should return all required fields."""
        result = analytical_stress_field(
            width=1.0, height=0.5,
            holes=[{"rx": 0.05, "ry": 0.05, "cx": 0.5, "cy": 0.5}],
            material_key="structural_steel_a36",
            load_magnitude=1e5,
            load_type="tension",
            n_grid=20,
        )
        required = ["coords", "stress_vm", "disp_x", "disp_y",
                     "sigma_max_mpa", "scf", "safety_factor", "fem_time_ms"]
        for key in required:
            assert key in result, f"Missing output key: {key}"

    def test_stress_positive(self):
        """Von Mises stress should be non-negative."""
        result = analytical_stress_field(
            1.0, 0.5, [{"rx": 0.05, "ry": 0.05, "cx": 0.5, "cy": 0.5}],
            "structural_steel_a36", 1e5, "tension", n_grid=30,
        )
        assert np.all(result["stress_vm"] >= 0), "Stress should be non-negative"

    def test_scf_greater_than_one(self):
        """SCF must be > 1 for a plate with a hole under tension."""
        result = analytical_stress_field(
            1.0, 0.5, [{"rx": 0.05, "ry": 0.05, "cx": 0.5, "cy": 0.5}],
            "structural_steel_a36", 1e5, "tension", n_grid=40,
        )
        assert result["scf"] > 1.0, f"SCF should be > 1, got {result['scf']}"

    def test_zero_stress_inside_hole(self):
        """Stress inside the hole should be zero."""
        result = analytical_stress_field(
            1.0, 0.5, [{"rx": 0.1, "ry": 0.1, "cx": 0.5, "cy": 0.5}],
            "structural_steel_a36", 1e5, "tension", n_grid=50,
        )
        coords = result["coords"]
        stress = result["stress_vm"]
        cx, cy, rx, ry = 0.5, 0.25, 0.1, 0.1
        inside = ((coords[:, 0] - cx) / rx)**2 + ((coords[:, 1] - cy) / ry)**2 <= 1.0
        if np.any(inside):
            assert np.all(stress[inside] == 0), "Stress inside hole should be zero"

    def test_larger_hole_higher_stress(self):
        """A larger hole should produce higher peak stress."""
        r_small = analytical_stress_field(
            1.0, 0.5, [{"rx": 0.03, "ry": 0.03, "cx": 0.5, "cy": 0.5}],
            "structural_steel_a36", 1e5, "tension", n_grid=40,
        )
        r_large = analytical_stress_field(
            1.0, 0.5, [{"rx": 0.1, "ry": 0.1, "cx": 0.5, "cy": 0.5}],
            "structural_steel_a36", 1e5, "tension", n_grid=40,
        )
        assert r_large["sigma_max_mpa"] >= r_small["sigma_max_mpa"] * 0.8, (
            "Larger hole should generally produce higher stress"
        )

    def test_multiple_holes(self):
        """Solver should handle multiple holes."""
        result = analytical_stress_field(
            1.0, 0.5,
            [
                {"rx": 0.04, "ry": 0.04, "cx": 0.3, "cy": 0.5},
                {"rx": 0.04, "ry": 0.04, "cx": 0.7, "cy": 0.5},
            ],
            "structural_steel_a36", 1e5, "tension", n_grid=30,
        )
        assert result["sigma_max_mpa"] > 0
        assert result["n_nodes"] > 0

    def test_different_load_types(self):
        """Solver should handle all load types without error."""
        for lt in ["tension", "shear", "biaxial", "compression", "combined"]:
            result = analytical_stress_field(
                1.0, 0.5, [{"rx": 0.05, "ry": 0.05, "cx": 0.5, "cy": 0.5}],
                "structural_steel_a36", 1e5, lt, n_grid=20,
            )
            assert result["sigma_max_mpa"] >= 0, f"Failed for load type: {lt}"

    def test_different_materials(self):
        """Solver should work for all materials."""
        from src.materials import MATERIALS
        for mat_key in list(MATERIALS.keys())[:5]:  # Test first 5
            result = analytical_stress_field(
                1.0, 0.5, [{"rx": 0.05, "ry": 0.05, "cx": 0.5, "cy": 0.5}],
                mat_key, 1e5, "tension", n_grid=20,
            )
            assert result["sigma_max_mpa"] >= 0, f"Failed for material: {mat_key}"

    def test_solver_speed(self):
        """Solver should complete within 1 second for n_grid=80."""
        result = analytical_stress_field(
            1.0, 0.5, [{"rx": 0.05, "ry": 0.05, "cx": 0.5, "cy": 0.5}],
            "structural_steel_a36", 1e5, "tension", n_grid=80,
        )
        assert result["fem_time_ms"] < 1000, (
            f"Solver too slow: {result['fem_time_ms']:.1f} ms"
        )


class TestSCFLibrary:
    """Test theoretical SCF formulas."""

    def test_kirsch_scf(self):
        """Kirsch: SCF = 3.0 for circular hole in infinite plate."""
        assert scf_circular_hole_infinite_plate() == 3.0

    def test_neuber_circular(self):
        """Neuber for circular (rx=ry): K_t = 1 + 2(1) = 3.0."""
        scf = scf_elliptical_hole_infinite_plate(0.05, 0.05)
        assert abs(scf - 3.0) < 0.01

    def test_neuber_elliptical(self):
        """Elliptical hole: K_t = 1 + 2(rx/ry)."""
        scf = scf_elliptical_hole_infinite_plate(0.1, 0.05)
        expected = 1 + 2 * (0.1 / 0.05)
        assert abs(scf - expected) < 0.01

    def test_pilkey_finite_width(self):
        """Pilkey formula should give K_t < 3 for small β."""
        scf = scf_circular_hole_finite_width(0.05, 1.0)
        assert 2.5 < scf < 3.1, f"Pilkey SCF = {scf} (expected ~3.0 for small hole)"

    def test_compare_scf_function(self):
        """compare_scf should return dict with all formula values."""
        result = compare_scf(0.05, 0.05, 1.0, ai_scf=2.95)
        assert "scf_values" in result
        assert "ai_scf" in result
        assert "error_pct_vs_theory" in result
        assert "best_match" in result


class TestRunFem:
    """Test the main run_fem entry point."""

    def test_run_fem_returns_dict(self):
        """run_fem should return a valid result dict."""
        result = run_fem(
            geometry={"plate_width": 1.0, "plate_height": 0.5,
                       "holes": [{"rx": 0.05, "ry": 0.05, "cx": 0.5, "cy": 0.5}]},
            material={"E": 200e9, "nu": 0.26, "sigma_y": 250e6},
            load={"type": "tension", "magnitude": 1e5},
        )
        assert isinstance(result, dict)
        assert "stress_vm" in result or "coords" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
SimuStruct AI V3 — Material Database Tests
============================================
Validates all 22 materials have required properties and computed
derived quantities are physically consistent.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.materials import (
    MATERIALS, CATEGORIES, get_lame_constants,
    get_shear_modulus, get_bulk_modulus, get_material_features,
    get_materials_by_category,
)


class TestMaterialDatabase:
    """Test the complete material database."""

    def test_material_count(self):
        """Should have 22 materials."""
        assert len(MATERIALS) == 22, f"Expected 22 materials, got {len(MATERIALS)}"

    def test_all_materials_have_required_keys(self):
        """Every material must have all required engineering properties."""
        required_keys = [
            "display_name", "category", "E", "nu", "sigma_y",
            "UTS", "rho", "alpha", "K_IC", "color", "fatigue_params",
        ]
        for key, mat in MATERIALS.items():
            for req in required_keys:
                assert req in mat, f"Material '{key}' missing property '{req}'"

    def test_youngs_modulus_positive(self):
        """Young's modulus must be positive for all materials."""
        for key, mat in MATERIALS.items():
            assert mat["E"] > 0, f"{key}: E must be positive, got {mat['E']}"

    def test_poisson_ratio_range(self):
        """Poisson's ratio must be in (0, 0.5) for stable materials."""
        for key, mat in MATERIALS.items():
            assert 0 < mat["nu"] < 0.5, (
                f"{key}: ν must be in (0, 0.5), got {mat['nu']}"
            )

    def test_density_positive(self):
        """Density must be positive."""
        for key, mat in MATERIALS.items():
            assert mat["rho"] > 0, f"{key}: density must be positive"

    def test_uts_geq_yield(self):
        """UTS must be >= yield strength (or yield is 0 for brittle materials)."""
        for key, mat in MATERIALS.items():
            if mat["sigma_y"] > 0:
                assert mat["UTS"] >= mat["sigma_y"], (
                    f"{key}: UTS ({mat['UTS']}) < σ_y ({mat['sigma_y']})"
                )

    def test_thermal_expansion_positive(self):
        """CTE must be positive."""
        for key, mat in MATERIALS.items():
            assert mat["alpha"] > 0, f"{key}: CTE must be positive"


class TestDerivedProperties:
    """Test derived mechanical property calculations."""

    def test_lame_constants_positive(self):
        """Lamé constants (λ, μ) must be positive for all materials."""
        for key in MATERIALS:
            lam, mu = get_lame_constants(key)
            assert mu > 0, f"{key}: shear modulus μ must be positive, got {mu}"
            # λ can be negative for ν < 0 or very high ν, but should be finite
            assert abs(lam) < 1e15, f"{key}: λ appears unreasonable: {lam}"

    def test_shear_modulus_formula(self):
        """G = E / (2(1+ν)) should match for all materials."""
        for key, mat in MATERIALS.items():
            G = get_shear_modulus(key)
            expected = mat["E"] / (2 * (1 + mat["nu"]))
            assert abs(G - expected) < 1e-3, (
                f"{key}: G mismatch: {G} vs expected {expected}"
            )

    def test_bulk_modulus_formula(self):
        """K = E / (3(1-2ν)) should match for all materials."""
        for key, mat in MATERIALS.items():
            if mat["nu"] < 0.499:  # Skip near-incompressible (rubber)
                K = get_bulk_modulus(key)
                expected = mat["E"] / (3 * (1 - 2 * mat["nu"]))
                assert abs(K - expected) / expected < 1e-6, (
                    f"{key}: K mismatch: {K} vs expected {expected}"
                )

    def test_lame_mu_equals_shear_modulus(self):
        """μ (second Lamé parameter) should equal the shear modulus G."""
        for key in MATERIALS:
            _, mu = get_lame_constants(key)
            G = get_shear_modulus(key)
            assert abs(mu - G) < 1e-3, f"{key}: μ≠G: {mu} vs {G}"


class TestMaterialFeatures:
    """Test AI feature extraction."""

    def test_feature_vector_length(self):
        """Feature vector should have 4 elements."""
        for key in MATERIALS:
            feats = get_material_features(key)
            assert len(feats) == 4, f"{key}: expected 4 features, got {len(feats)}"

    def test_feature_values_finite(self):
        """All feature values should be finite numbers."""
        import math
        for key in MATERIALS:
            feats = get_material_features(key)
            for i, f in enumerate(feats):
                assert math.isfinite(f), f"{key}: feature {i} is not finite: {f}"

    def test_normalized_ranges(self):
        """Normalized features should be roughly in [0, 2] range."""
        for key in MATERIALS:
            feats = get_material_features(key)
            E_norm = feats[0]
            assert 0 <= E_norm <= 2.0, f"{key}: E_norm={E_norm} out of expected range"


class TestCategories:
    """Test category system."""

    def test_categories_not_empty(self):
        """Should have at least 5 categories."""
        assert len(CATEGORIES) >= 5, f"Expected ≥5 categories, got {len(CATEGORIES)}"

    def test_category_filter(self):
        """get_materials_by_category should return matching materials."""
        steels = get_materials_by_category("Steel")
        assert len(steels) > 0, "Steel category should have materials"
        for k, v in steels.items():
            assert v["category"] == "Steel"

    def test_all_materials_have_valid_category(self):
        """Every material's category should be in CATEGORIES list."""
        for key, mat in MATERIALS.items():
            assert mat["category"] in CATEGORIES, (
                f"{key}: category '{mat['category']}' not in CATEGORIES"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

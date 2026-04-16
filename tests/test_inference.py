"""
SimuStruct AI V3 — AI Inference Tests
=======================================
Tests model architecture, output shapes, and inference pipeline.
"""

import pytest
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.enhanced_mlp import EnhancedMLP, FourierEncoding, build_enhanced_mlp


class TestFourierEncoding:
    """Test Fourier positional encoding module."""

    def test_output_shape(self):
        """Output should be (N, 2 * 2 * n_freqs)."""
        enc = FourierEncoding(n_freqs=8, input_dim=2)
        x = torch.randn(100, 2)
        out = enc(x)
        assert out.shape == (100, 32), f"Expected (100, 32), got {out.shape}"

    def test_output_dim_property(self):
        """output_dim property should match actual output."""
        enc = FourierEncoding(n_freqs=8, input_dim=2)
        assert enc.output_dim == 32

    def test_different_frequencies(self):
        """Different n_freqs should produce different output sizes."""
        for nf in [4, 8, 16]:
            enc = FourierEncoding(n_freqs=nf, input_dim=2)
            x = torch.randn(10, 2)
            out = enc(x)
            assert out.shape[1] == 2 * 2 * nf

    def test_deterministic(self):
        """Same input should produce same output."""
        enc = FourierEncoding(n_freqs=8)
        x = torch.randn(10, 2)
        out1 = enc(x)
        out2 = enc(x)
        assert torch.allclose(out1, out2)


class TestEnhancedMLP:
    """Test the Enhanced MLP architecture."""

    @pytest.fixture
    def model(self):
        return build_enhanced_mlp()

    @pytest.fixture
    def batch_inputs(self):
        batch = 256
        return {
            "coords": torch.randn(batch, 2),
            "material_feats": torch.randn(batch, 4),
            "geo_feats": torch.randn(batch, 4),
            "load_feats": torch.randn(batch, 5),
        }

    def test_output_shape(self, model, batch_inputs):
        """Output should be (N, 3) for [σ_vm, disp_x, disp_y]."""
        out = model(**batch_inputs)
        assert out.shape == (256, 3), f"Expected (256, 3), got {out.shape}"

    def test_parameter_count(self, model):
        """Model should have a reasonable number of parameters."""
        n_params = model.count_parameters()
        assert 10000 < n_params < 1000000, (
            f"Parameter count {n_params} seems unreasonable"
        )

    def test_gradient_flow(self, model, batch_inputs):
        """Gradients should flow through all parameters."""
        out = model(**batch_inputs)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_different_batch_sizes(self, model):
        """Model should handle different batch sizes."""
        for batch in [1, 16, 128, 1024]:
            coords = torch.randn(batch, 2)
            mat = torch.randn(batch, 4)
            geo = torch.randn(batch, 4)
            load = torch.randn(batch, 5)
            out = model(coords, mat, geo, load)
            assert out.shape == (batch, 3)

    def test_without_load_features(self, model):
        """Model should work without load features (optional)."""
        coords = torch.randn(32, 2)
        mat = torch.randn(32, 4)
        geo = torch.randn(32, 4)
        out = model(coords, mat, geo)
        assert out.shape == (32, 3)

    def test_custom_hidden_dims(self):
        """Custom hidden dimensions should work."""
        model = build_enhanced_mlp({"hidden_dims": [64, 128, 64]})
        coords = torch.randn(16, 2)
        mat = torch.randn(16, 4)
        geo = torch.randn(16, 4)
        load = torch.randn(16, 5)
        out = model(coords, mat, geo, load)
        assert out.shape == (16, 3)

    def test_eval_mode(self, model, batch_inputs):
        """Model should work in eval mode (no dropout)."""
        model.eval()
        with torch.no_grad():
            out = model(**batch_inputs)
        assert out.shape == (256, 3)
        assert not torch.any(torch.isnan(out)), "NaN in eval mode output"

    def test_output_finite(self, model, batch_inputs):
        """All output values should be finite."""
        model.eval()
        with torch.no_grad():
            out = model(**batch_inputs)
        assert torch.all(torch.isfinite(out)), "Non-finite values in output"


class TestMCDropout:
    """Test Monte Carlo Dropout model."""

    def test_import(self):
        """MC Dropout module should be importable."""
        from models.mc_dropout import MCDropoutMLP, predict_with_uncertainty

    def test_mc_dropout_output(self):
        """MC Dropout should produce mean and std."""
        from models.mc_dropout import MCDropoutMLP, predict_with_uncertainty

        model = MCDropoutMLP(dropout_p=0.1)
        coords = torch.randn(64, 2)
        mat = torch.randn(64, 4)
        geo = torch.randn(64, 4)
        load = torch.randn(64, 5)

        mean, std, all_preds = predict_with_uncertainty(
            model, coords, mat, geo, load, n_samples=10
        )
        assert mean.shape == (64, 3)
        assert std.shape == (64, 3)
        assert all_preds.shape == (10, 64, 3)

    def test_uncertainty_positive(self):
        """Uncertainty should be non-negative."""
        from models.mc_dropout import MCDropoutMLP, predict_with_uncertainty

        model = MCDropoutMLP(dropout_p=0.2)
        coords = torch.randn(32, 2)
        mat = torch.randn(32, 4)
        geo = torch.randn(32, 4)

        _, std, _ = predict_with_uncertainty(model, coords, mat, geo, n_samples=20)
        assert torch.all(std >= 0), "Uncertainty should be non-negative"


class TestFatigueModule:
    """Test fatigue estimation module."""

    def test_fatigue_life(self):
        """Fatigue life should return positive cycles."""
        from src.fatigue import estimate_fatigue_life
        result = estimate_fatigue_life(200.0, 0.0, "structural_steel_a36")
        assert result["cycles_to_failure"] > 0
        assert result["life_category"] in [
            "Infinite Life", "High-Cycle Fatigue (HCF)",
            "Low-Cycle Fatigue (LCF)", "Static Failure Risk"
        ]

    def test_sn_curve(self):
        """S-N curve should return valid data."""
        from src.fatigue import generate_sn_curve
        sn = generate_sn_curve("aluminum_6061_t6", n_points=50)
        assert len(sn["N"]) == 50
        assert len(sn["S"]) == 50
        assert all(n > 0 for n in sn["N"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

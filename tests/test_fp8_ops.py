"""Unit tests for FP8 simulation utilities.

These tests run on CPU without any model download.
"""

import pytest
import torch

from utils.fp8_ops import simulate_fp8, _manual_fp8_round


class TestSimulateFP8:
    """Tests for simulate_fp8."""

    def test_output_dtype_preserved(self):
        """Output should keep the original dtype."""
        for dtype in [torch.float32, torch.bfloat16]:
            x = torch.randn(4, 8, dtype=dtype)
            y = simulate_fp8(x)
            assert y.dtype == dtype

    def test_output_shape_preserved(self):
        """Output shape must match input shape."""
        x = torch.randn(2, 3, 16)
        y = simulate_fp8(x)
        assert y.shape == x.shape

    def test_introduces_precision_loss(self):
        """FP8 rounding should NOT be an identity op on general inputs."""
        torch.manual_seed(0)
        x = torch.randn(64, 64)
        y = simulate_fp8(x)
        # Should be close but not identical
        assert not torch.equal(x, y), "FP8 simulation should introduce precision loss"
        assert torch.allclose(x, y, atol=0.5), "FP8 result should stay reasonably close"

    def test_small_values_survive(self):
        """Very small values should not all collapse to zero."""
        x = torch.tensor([1e-4, 1e-3, 1e-2, 1e-1])
        y = simulate_fp8(x)
        assert (y.abs() > 0).all(), "Small values should survive FP8 quantisation"

    def test_large_values_survive(self):
        """Large values within FP8 range should be preserved approximately."""
        x = torch.tensor([100.0, 200.0, 300.0, 400.0])
        y = simulate_fp8(x)
        assert torch.allclose(x, y, rtol=0.15), "Large values should be close after FP8"

    def test_zero_tensor(self):
        """Empty or zero tensors should pass through safely."""
        z = torch.zeros(0)
        assert simulate_fp8(z).numel() == 0

        z = torch.zeros(4, 4)
        y = simulate_fp8(z)
        assert (y == 0).all()

    def test_single_element(self):
        x = torch.tensor([42.0])
        y = simulate_fp8(x)
        assert y.shape == x.shape
        assert abs(y.item() - x.item()) < 5.0


class TestManualFP8Round:
    """Tests for the fallback manual rounding."""

    def test_precision_reduction(self):
        torch.manual_seed(1)
        x = torch.randn(32, 32)
        y = _manual_fp8_round(x)
        assert not torch.equal(x, y)
        assert torch.allclose(x, y, atol=1.0)

    def test_signs_preserved(self):
        x = torch.tensor([-3.0, -1.0, 0.5, 2.0])
        y = _manual_fp8_round(x)
        assert (x.sign() == y.sign()).all()

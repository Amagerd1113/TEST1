"""
Unit tests for GR field module.
"""

import pytest
import torch
from src.core.gr_field import AdaptiveGRFieldManager


@pytest.mark.unit
class TestAdaptiveGRFieldManager:
    """Test suite for AdaptiveGRFieldManager."""

    def test_module_initialization(self, test_config):
        """Test that the module initializes correctly."""
        module = AdaptiveGRFieldManager(test_config)
        assert module is not None

    def test_field_computation(self, test_config, device):
        """Test GR field computation."""
        module = AdaptiveGRFieldManager(test_config).to(device)
        module.eval()

        batch_size = 2
        affordance_field = torch.randn(batch_size, 32, 32, 16, device=device)
        position = torch.randn(batch_size, 3, device=device)
        velocity = torch.randn(batch_size, 3, device=device)
        field_coupling = torch.tensor([8 * 3.14159, 1.0, 0.1], device=device)

        with torch.no_grad():
            output = module(
                affordance_field=affordance_field,
                position=position,
                velocity=velocity,
                field_coupling=field_coupling,
                previous_field=None
            )

        # Check output keys
        assert 'metric_tensor' in output
        assert 'christoffel_symbols' in output
        assert 'riemann_curvature' in output

    def test_metric_tensor_properties(self, test_config, device):
        """Test that metric tensor has correct properties."""
        module = AdaptiveGRFieldManager(test_config).to(device)
        module.eval()

        batch_size = 2
        affordance_field = torch.randn(batch_size, 32, 32, 16, device=device)
        position = torch.randn(batch_size, 3, device=device)
        velocity = torch.randn(batch_size, 3, device=device)
        field_coupling = torch.tensor([8 * 3.14159, 1.0, 0.1], device=device)

        with torch.no_grad():
            output = module(
                affordance_field=affordance_field,
                position=position,
                velocity=velocity,
                field_coupling=field_coupling,
                previous_field=None
            )

        metric = output['metric_tensor']

        # Metric tensor should have spatial dimensions
        assert len(metric.shape) == 4  # [B, H, W, components]
        assert metric.shape[0] == batch_size

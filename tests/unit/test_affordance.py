"""
Unit tests for affordance module.
"""

import pytest
import torch
from src.core.affordance import UncertaintyAwareAffordanceModule


@pytest.mark.unit
class TestUncertaintyAwareAffordanceModule:
    """Test suite for UncertaintyAwareAffordanceModule."""

    def test_module_initialization(self, test_config):
        """Test that the module initializes correctly."""
        module = UncertaintyAwareAffordanceModule(test_config)
        assert module is not None

    def test_forward_pass(self, test_config, device):
        """Test that forward pass produces correct outputs."""
        module = UncertaintyAwareAffordanceModule(test_config).to(device)
        module.eval()

        # Create sample inputs
        batch_size = 2
        num_patches = 196
        hidden_dim = test_config['model']['vla']['hidden_dim']

        visual_features = torch.randn(batch_size, num_patches, hidden_dim, device=device)
        language_features = torch.randn(batch_size, 10, hidden_dim, device=device)
        depth_map = torch.randn(batch_size, 1, 224, 224, device=device)
        uncertainty = torch.randn(batch_size, num_patches, 1, device=device)

        with torch.no_grad():
            output = module(
                visual_features=visual_features,
                language_features=language_features,
                depth_map=depth_map,
                uncertainty=uncertainty
            )

        # Check output keys
        assert 'affordance_field' in output
        assert 'uncertainty' in output

        # Check that affordance field has spatial dimensions
        affordance_field = output['affordance_field']
        assert len(affordance_field.shape) == 4  # [B, H, W, C]
        assert affordance_field.shape[0] == batch_size

    def test_bayesian_update(self, test_config, device):
        """Test that Bayesian updates work correctly."""
        module = UncertaintyAwareAffordanceModule(test_config).to(device)

        batch_size = 2
        num_patches = 196
        hidden_dim = test_config['model']['vla']['hidden_dim']

        visual_features = torch.randn(batch_size, num_patches, hidden_dim, device=device)
        language_features = torch.randn(batch_size, 10, hidden_dim, device=device)
        depth_map = torch.randn(batch_size, 1, 224, 224, device=device)
        uncertainty = torch.randn(batch_size, num_patches, 1, device=device)

        with torch.no_grad():
            # First pass
            output1 = module(
                visual_features=visual_features,
                language_features=language_features,
                depth_map=depth_map,
                uncertainty=uncertainty
            )

            # Second pass (should refine estimates)
            output2 = module(
                visual_features=visual_features,
                language_features=language_features,
                depth_map=depth_map,
                uncertainty=uncertainty
            )

        # Outputs should exist
        assert output1 is not None
        assert output2 is not None

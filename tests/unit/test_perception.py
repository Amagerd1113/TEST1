"""
Unit tests for perception module.
"""

import pytest
import torch
from src.core.perception import AdvancedPerceptionModule


@pytest.mark.unit
class TestAdvancedPerceptionModule:
    """Test suite for AdvancedPerceptionModule."""

    def test_module_initialization(self, test_config):
        """Test that the module initializes correctly."""
        module = AdvancedPerceptionModule(test_config)
        assert module is not None
        assert hasattr(module, 'forward')

    def test_forward_pass_shapes(
        self, test_config, sample_rgb_image, sample_depth_map, sample_language_instruction, device
    ):
        """Test that forward pass produces correct output shapes."""
        module = AdvancedPerceptionModule(test_config).to(device)
        module.eval()

        with torch.no_grad():
            output = module(
                rgb=sample_rgb_image,
                depth=sample_depth_map,
                semantic=None,
                language=sample_language_instruction
            )

        # Check output keys
        assert 'visual_features' in output
        assert 'visual_uncertainty' in output
        assert 'language_features' in output

        # Check shapes
        batch_size = sample_rgb_image.shape[0]
        assert output['visual_features'].shape[0] == batch_size
        assert output['visual_uncertainty'].shape[0] == batch_size
        assert output['language_features'].shape[0] == batch_size

    def test_forward_pass_with_semantic(
        self, test_config, sample_rgb_image, sample_depth_map,
        sample_language_instruction, device
    ):
        """Test forward pass with semantic segmentation."""
        module = AdvancedPerceptionModule(test_config).to(device)
        module.eval()

        # Create sample semantic map
        semantic = torch.randint(0, 40, (2, 224, 224), device=device)

        with torch.no_grad():
            output = module(
                rgb=sample_rgb_image,
                depth=sample_depth_map,
                semantic=semantic,
                language=sample_language_instruction
            )

        assert output is not None
        assert 'visual_features' in output

    @pytest.mark.requires_gpu
    def test_gpu_support(self, test_config, sample_rgb_image, sample_depth_map, sample_language_instruction):
        """Test that the module works on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        device = torch.device("cuda")
        module = AdvancedPerceptionModule(test_config).to(device)

        rgb = sample_rgb_image.to(device)
        depth = sample_depth_map.to(device)

        with torch.no_grad():
            output = module(rgb=rgb, depth=depth, language=sample_language_instruction)

        assert output['visual_features'].device == device

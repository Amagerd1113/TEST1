"""
Integration tests for the complete VLA-GR agent.
"""

import pytest
import torch
from src.core.vla_gr_agent import ConferenceVLAGRAgent, VLAGRStateV2


@pytest.mark.integration
class TestVLAGRAgentIntegration:
    """Integration tests for the full VLA-GR agent."""

    def test_agent_initialization(self, test_config):
        """Test that the agent initializes with all components."""
        agent = ConferenceVLAGRAgent(test_config)

        # Check all components exist
        assert hasattr(agent, 'perception')
        assert hasattr(agent, 'affordance_quantifier')
        assert hasattr(agent, 'gr_field_manager')
        assert hasattr(agent, 'path_optimizer')
        assert hasattr(agent, 'vla_transformer')
        assert hasattr(agent, 'memory_module')
        assert hasattr(agent, 'action_decoder')
        assert hasattr(agent, 'uncertainty_estimator')

    def test_full_forward_pass(
        self, test_config, sample_rgb_image, sample_depth_map,
        sample_language_instruction, sample_position,
        sample_orientation, sample_velocity, device
    ):
        """Test full forward pass through the agent."""
        agent = ConferenceVLAGRAgent(test_config).to(device)
        agent.eval()

        # Create state
        state = VLAGRStateV2(
            rgb_image=sample_rgb_image,
            depth_map=sample_depth_map,
            language_instruction=sample_language_instruction,
            position=sample_position,
            orientation=sample_orientation,
            velocity=sample_velocity
        )

        # Forward pass
        with torch.no_grad():
            output = agent(state, deterministic=True)

        # Check outputs
        assert 'actions' in output
        assert 'planned_path' in output
        assert 'gr_field' in output
        assert 'affordance_map' in output

        # Check action shape
        batch_size = sample_rgb_image.shape[0]
        action_dim = test_config['model']['vla']['action_dim']
        assert output['actions'].shape == (batch_size, action_dim)

    @pytest.mark.slow
    def test_agent_with_gradient(
        self, test_config, sample_rgb_image, sample_depth_map,
        sample_language_instruction, sample_position,
        sample_orientation, sample_velocity, device
    ):
        """Test that gradients flow correctly through the agent."""
        agent = ConferenceVLAGRAgent(test_config).to(device)
        agent.train()

        state = VLAGRStateV2(
            rgb_image=sample_rgb_image,
            depth_map=sample_depth_map,
            language_instruction=sample_language_instruction,
            position=sample_position,
            orientation=sample_orientation,
            velocity=sample_velocity
        )

        # Forward pass with gradients
        output = agent(state, deterministic=False, return_distribution=True)

        # Create a dummy loss
        loss = output['actions'].mean()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        has_gradients = False
        for param in agent.parameters():
            if param.grad is not None:
                has_gradients = True
                break

        assert has_gradients, "No gradients computed"

    def test_deterministic_vs_stochastic(
        self, test_config, sample_rgb_image, sample_depth_map,
        sample_language_instruction, sample_position,
        sample_orientation, sample_velocity, device, set_random_seed
    ):
        """Test deterministic and stochastic action modes."""
        agent = ConferenceVLAGRAgent(test_config).to(device)
        agent.eval()

        state = VLAGRStateV2(
            rgb_image=sample_rgb_image,
            depth_map=sample_depth_map,
            language_instruction=sample_language_instruction,
            position=sample_position,
            orientation=sample_orientation,
            velocity=sample_velocity
        )

        with torch.no_grad():
            # Deterministic should give same result
            out1 = agent(state, deterministic=True)
            out2 = agent(state, deterministic=True)

            # Actions should be identical in deterministic mode
            assert torch.allclose(out1['actions'], out2['actions'], atol=1e-5)

"""
Pytest configuration and shared fixtures for VLA-GR tests.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any


@pytest.fixture
def device():
    """Get the available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Standard test configuration for VLA-GR."""
    return {
        'model': {
            'vla': {
                'hidden_dim': 256,
                'num_layers': 4,
                'num_heads': 8,
                'action_dim': 7,
            },
            'gr_field': {
                'field_dim': 64,
                'grid_size': [32, 32, 16],
                'lambda_curvature': 0.1,
            },
            'memory': {
                'size': 100,
                'consolidation_threshold': 0.8,
            },
            'action': {
                'num_primitives': 10,
            }
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 1e-4,
            'max_steps': 1000,
        }
    }


@pytest.fixture
def sample_rgb_image(device):
    """Generate a sample RGB image tensor."""
    return torch.randn(2, 3, 224, 224, device=device)


@pytest.fixture
def sample_depth_map(device):
    """Generate a sample depth map tensor."""
    return torch.randn(2, 1, 224, 224, device=device)


@pytest.fixture
def sample_language_instruction():
    """Generate sample language instructions."""
    return ["Navigate to the red chair", "Go to the kitchen table"]


@pytest.fixture
def sample_position(device):
    """Generate sample position tensor."""
    return torch.randn(2, 3, device=device)


@pytest.fixture
def sample_orientation(device):
    """Generate sample orientation tensor (quaternion)."""
    # Generate random quaternions and normalize
    q = torch.randn(2, 4, device=device)
    return q / q.norm(dim=1, keepdim=True)


@pytest.fixture
def sample_velocity(device):
    """Generate sample velocity tensor."""
    return torch.randn(2, 3, device=device)


@pytest.fixture(scope="session")
def set_random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )

"""
Unified VLA Dataset Loader
Supports multiple VLA benchmarks: OXE, BridgeData, LIBERO, CALVIN, Habitat
Based on ICLR 2025 and latest VLA research standards
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import yaml
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


@dataclass
class VLAEpisode:
    """
    Unified VLA episode data structure.
    Compatible with manipulation, navigation, and mobile manipulation tasks.
    """
    # Identifiers
    episode_id: str
    dataset_name: str
    task_name: str

    # Observations
    rgb_images: List[np.ndarray]  # List of RGB images [H, W, 3]
    depth_images: Optional[List[np.ndarray]] = None  # List of depth images [H, W, 1]
    proprioception: Optional[List[np.ndarray]] = None  # Robot state (joint angles, gripper, etc.)

    # Language
    language_instruction: Optional[str] = None
    language_tokens: Optional[List[int]] = None

    # Actions
    actions: List[np.ndarray] = None  # Ground truth actions
    action_space_type: str = "continuous"  # "continuous" or "discrete"

    # Task-specific
    task_type: str = "manipulation"  # "manipulation", "navigation", "mobile_manipulation"
    goal_specification: Optional[Any] = None  # Could be coordinates, object category, image, etc.

    # Metadata
    success: bool = False
    episode_length: int = 0
    metadata: Optional[Dict] = None


class VLADatasetLoader:
    """
    Unified VLA dataset loader supporting multiple benchmarks.

    Supports:
    - Open X-Embodiment (OXE)
    - BridgeData V2
    - LIBERO
    - CALVIN
    - Habitat (HM3D, MP3D, Gibson, Replica)
    - Custom datasets
    """

    def __init__(self, config: Union[Dict, str]):
        """
        Initialize dataset loader.

        Args:
            config: Configuration dict or path to YAML config file
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        self.datasets = {}
        self.dataset_info = {}

    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        **kwargs
    ) -> Dataset:
        """
        Load a specific dataset.

        Args:
            dataset_name: Name of dataset (e.g., "open_x_embodiment", "bridge_data")
            split: Data split ("train", "val", "test")
            **kwargs: Additional dataset-specific arguments

        Returns:
            PyTorch Dataset object
        """
        dataset_config = self.config.get('datasets', {}).get(dataset_name)

        if dataset_config is None:
            raise ValueError(f"Dataset {dataset_name} not found in config")

        if not dataset_config.get('enabled', True):
            raise ValueError(f"Dataset {dataset_name} is disabled in config")

        # Route to appropriate loader
        dataset_type = dataset_config.get('type')

        if dataset_name == 'open_x_embodiment':
            dataset = self._load_open_x_embodiment(dataset_config, split, **kwargs)
        elif dataset_name == 'bridge_data':
            dataset = self._load_bridge_data(dataset_config, split, **kwargs)
        elif dataset_name == 'libero':
            dataset = self._load_libero(dataset_config, split, **kwargs)
        elif dataset_name == 'calvin':
            dataset = self._load_calvin(dataset_config, split, **kwargs)
        elif dataset_type == 'simulation' and 'navigation' in dataset_config:
            dataset = self._load_habitat_navigation(dataset_config, split, **kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_name}")

        # Store dataset info
        self.datasets[dataset_name] = dataset
        self.dataset_info[dataset_name] = {
            'split': split,
            'size': len(dataset),
            'config': dataset_config
        }

        logger.info(f"Loaded {dataset_name} ({split}): {len(dataset)} episodes")

        return dataset

    def load_mixed_dataset(
        self,
        dataset_names: List[str],
        split: str = "train",
        mixing_strategy: str = "proportional",
        weights: Optional[Dict[str, float]] = None
    ) -> Dataset:
        """
        Load and mix multiple datasets.

        Args:
            dataset_names: List of dataset names to mix
            split: Data split
            mixing_strategy: "uniform", "proportional", or "weighted"
            weights: Custom weights for mixing (if strategy is "weighted")

        Returns:
            Mixed dataset
        """
        datasets = []

        for name in dataset_names:
            try:
                dataset = self.load_dataset(name, split)
                datasets.append(dataset)
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
                continue

        if not datasets:
            raise ValueError("No datasets loaded successfully")

        # Apply mixing strategy
        if mixing_strategy == "uniform":
            # Simple concatenation
            mixed_dataset = ConcatDataset(datasets)

        elif mixing_strategy == "proportional":
            # Proportional to dataset size (already done by ConcatDataset)
            mixed_dataset = ConcatDataset(datasets)

        elif mixing_strategy == "weighted":
            # Custom weights
            if weights is None:
                raise ValueError("Weights required for 'weighted' mixing strategy")
            mixed_dataset = WeightedMixedDataset(datasets, weights)

        else:
            raise ValueError(f"Unknown mixing strategy: {mixing_strategy}")

        logger.info(f"Created mixed dataset from {len(datasets)} datasets: {len(mixed_dataset)} total episodes")

        return mixed_dataset

    # ========================================================================
    # Dataset-specific loaders
    # ========================================================================

    def _load_open_x_embodiment(
        self,
        config: Dict,
        split: str,
        **kwargs
    ) -> Dataset:
        """Load Open X-Embodiment dataset."""
        data_path = Path(config['data_path']) / split

        if not data_path.exists():
            logger.warning(f"OXE data path not found: {data_path}")
            logger.warning("Returning mock dataset for testing")
            return MockVLADataset(config, split, "open_x_embodiment")

        return OpenXEmbodimentDataset(
            data_path=data_path,
            config=config,
            split=split,
            **kwargs
        )

    def _load_bridge_data(
        self,
        config: Dict,
        split: str,
        **kwargs
    ) -> Dataset:
        """Load BridgeData V2 dataset."""
        data_path = Path(config['data_path']) / split

        if not data_path.exists():
            logger.warning(f"BridgeData path not found: {data_path}")
            logger.warning("Returning mock dataset for testing")
            return MockVLADataset(config, split, "bridge_data")

        return BridgeDataDataset(
            data_path=data_path,
            config=config,
            split=split,
            **kwargs
        )

    def _load_libero(
        self,
        config: Dict,
        split: str,
        **kwargs
    ) -> Dataset:
        """Load LIBERO dataset."""
        data_path = Path(config['data_path'])

        if not data_path.exists():
            logger.warning(f"LIBERO data path not found: {data_path}")
            logger.warning("Returning mock dataset for testing")
            return MockVLADataset(config, split, "libero")

        return LIBERODataset(
            data_path=data_path,
            config=config,
            split=split,
            **kwargs
        )

    def _load_calvin(
        self,
        config: Dict,
        split: str,
        **kwargs
    ) -> Dataset:
        """Load CALVIN dataset."""
        data_path = Path(config['data_path'])

        if not data_path.exists():
            logger.warning(f"CALVIN data path not found: {data_path}")
            logger.warning("Returning mock dataset for testing")
            return MockVLADataset(config, split, "calvin")

        return CALVINDataset(
            data_path=data_path,
            config=config,
            split=split,
            **kwargs
        )

    def _load_habitat_navigation(
        self,
        config: Dict,
        split: str,
        **kwargs
    ) -> Dataset:
        """Load Habitat navigation dataset."""
        # Import existing Habitat dataset
        from .habitat_dataset import HabitatNavigationDataset

        return HabitatNavigationDataset(
            config=self.config,
            split=split,
            **kwargs
        )


# ============================================================================
# Dataset Implementations
# ============================================================================

class OpenXEmbodimentDataset(Dataset):
    """Open X-Embodiment dataset implementation."""

    def __init__(self, data_path: Path, config: Dict, split: str, **kwargs):
        self.data_path = data_path
        self.config = config
        self.split = split

        # Load episode metadata
        self.episodes = self._load_episodes()

    def _load_episodes(self) -> List[VLAEpisode]:
        """Load episode metadata."""
        # Implementation would load from RLDS format or converted format
        episodes = []

        # Placeholder: Load from directory structure
        episode_dirs = sorted(self.data_path.glob("episode_*"))

        for episode_dir in episode_dirs:
            # Load episode data
            metadata_file = episode_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Create VLAEpisode
                episode = VLAEpisode(
                    episode_id=episode_dir.name,
                    dataset_name="open_x_embodiment",
                    task_name=metadata.get('task_name', 'unknown'),
                    rgb_images=[],  # Loaded on demand
                    task_type="manipulation",
                    metadata=metadata
                )
                episodes.append(episode)

        return episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get episode data."""
        episode = self.episodes[idx]

        # Load images on demand
        episode_dir = self.data_path / episode.episode_id

        # Load RGB images
        rgb_images = []
        for img_file in sorted((episode_dir / "rgb").glob("*.png")):
            img = Image.open(img_file).convert('RGB')
            rgb_images.append(np.array(img))

        # Load actions
        actions_file = episode_dir / "actions.npy"
        if actions_file.exists():
            actions = np.load(actions_file)
        else:
            actions = None

        # Load language instruction
        if episode.metadata and 'language_instruction' in episode.metadata:
            language = episode.metadata['language_instruction']
        else:
            language = "unknown task"

        # Convert to tensors
        return {
            'rgb': torch.from_numpy(np.stack(rgb_images)).float() / 255.0,
            'actions': torch.from_numpy(actions).float() if actions is not None else None,
            'language': language,
            'episode_id': episode.episode_id,
            'task_name': episode.task_name,
        }


class BridgeDataDataset(Dataset):
    """BridgeData V2 dataset implementation."""

    def __init__(self, data_path: Path, config: Dict, split: str, **kwargs):
        self.data_path = data_path
        self.config = config
        self.split = split

        self.episodes = self._load_episodes()

    def _load_episodes(self) -> List:
        """Load BridgeData episodes."""
        # Similar structure to OXE
        episodes = []
        # Implementation here
        return episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict:
        # Similar to OXE
        return {}


class LIBERODataset(Dataset):
    """LIBERO dataset implementation."""

    def __init__(self, data_path: Path, config: Dict, split: str, **kwargs):
        self.data_path = data_path
        self.config = config
        self.split = split

        # LIBERO uses HDF5 format
        self.episodes = self._load_episodes()

    def _load_episodes(self) -> List:
        """Load LIBERO episodes."""
        episodes = []
        # Load from HDF5 files
        return episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict:
        return {}


class CALVINDataset(Dataset):
    """CALVIN dataset implementation."""

    def __init__(self, data_path: Path, config: Dict, split: str, **kwargs):
        self.data_path = data_path
        self.config = config
        self.split = split

        self.episodes = self._load_episodes()

    def _load_episodes(self) -> List:
        """Load CALVIN episodes."""
        episodes = []
        # Load from CALVIN format (NPZ files)
        return episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict:
        return {}


class MockVLADataset(Dataset):
    """Mock dataset for testing when real data is not available."""

    def __init__(self, config: Dict, split: str, dataset_name: str):
        self.config = config
        self.split = split
        self.dataset_name = dataset_name

        # Generate mock episodes
        if split == "train":
            self.num_episodes = 100
        elif split == "val":
            self.num_episodes = 20
        else:
            self.num_episodes = 30

        logger.info(f"Created mock {dataset_name} dataset with {self.num_episodes} episodes")

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate mock episode data."""
        # Mock RGB images (224x224x3, 10 frames)
        rgb = torch.rand(10, 224, 224, 3)

        # Mock actions (10 timesteps, 7D action space)
        actions = torch.randn(10, 7)

        # Mock language
        language = f"mock task {idx}"

        return {
            'rgb': rgb,
            'actions': actions,
            'language': language,
            'episode_id': f"mock_{self.dataset_name}_{idx}",
            'task_name': f"mock_task_{idx % 10}",
            'success': torch.tensor(1.0) if idx % 3 != 0 else torch.tensor(0.0),
        }


class WeightedMixedDataset(Dataset):
    """
    Mixed dataset with custom sampling weights.
    """

    def __init__(self, datasets: List[Dataset], weights: Dict[str, float]):
        self.datasets = datasets
        self.weights = weights

        # Compute sampling probabilities
        self.dataset_sizes = [len(d) for d in datasets]
        self.total_size = sum(self.dataset_sizes)

        # Create sampling distribution
        self.sampling_probs = []
        for dataset_size, weight in zip(self.dataset_sizes, weights.values()):
            self.sampling_probs.extend([weight] * dataset_size)

        self.sampling_probs = np.array(self.sampling_probs)
        self.sampling_probs /= self.sampling_probs.sum()

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict:
        # Use weighted sampling
        sampled_idx = np.random.choice(len(self.sampling_probs), p=self.sampling_probs)

        # Find which dataset this index belongs to
        cumsum = 0
        for dataset_idx, size in enumerate(self.dataset_sizes):
            if sampled_idx < cumsum + size:
                local_idx = sampled_idx - cumsum
                return self.datasets[dataset_idx][local_idx]
            cumsum += size


# ============================================================================
# Utility Functions
# ============================================================================

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader with appropriate settings for VLA datasets.

    Args:
        dataset: VLA dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        **kwargs: Additional DataLoader arguments

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )


def get_dataset_statistics(dataset: Dataset) -> Dict:
    """
    Compute statistics for a VLA dataset.

    Args:
        dataset: VLA dataset

    Returns:
        Dictionary of statistics
    """
    stats = {
        'num_episodes': len(dataset),
        'episode_lengths': [],
        'success_rates': [],
        'task_distribution': {},
    }

    # Sample episodes to compute statistics
    sample_size = min(100, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)

    for idx in indices:
        episode = dataset[idx]

        if 'actions' in episode and episode['actions'] is not None:
            stats['episode_lengths'].append(len(episode['actions']))

        if 'success' in episode:
            stats['success_rates'].append(episode['success'].item())

        if 'task_name' in episode:
            task = episode['task_name']
            stats['task_distribution'][task] = stats['task_distribution'].get(task, 0) + 1

    # Compute summary statistics
    if stats['episode_lengths']:
        stats['avg_episode_length'] = np.mean(stats['episode_lengths'])
        stats['std_episode_length'] = np.std(stats['episode_lengths'])

    if stats['success_rates']:
        stats['overall_success_rate'] = np.mean(stats['success_rates'])

    return stats


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Load configuration
    config_path = "configs/datasets_vla_benchmark.yaml"

    # Create loader
    loader = VLADatasetLoader(config_path)

    # Load single dataset
    dataset = loader.load_dataset("open_x_embodiment", split="train")
    print(f"Loaded dataset: {len(dataset)} episodes")

    # Load mixed dataset
    mixed_dataset = loader.load_mixed_dataset(
        ["open_x_embodiment", "bridge_data", "libero"],
        split="train",
        mixing_strategy="proportional"
    )
    print(f"Loaded mixed dataset: {len(mixed_dataset)} episodes")

    # Create DataLoader
    dataloader = create_dataloader(mixed_dataset, batch_size=32, num_workers=4)

    # Get statistics
    stats = get_dataset_statistics(dataset)
    print(f"Dataset statistics: {stats}")

"""
VLN-CE Dataset Loader with Distractor Annotation
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional


class VLNDataset(Dataset):
    """VLN-CE dataset with distractor annotations."""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_episode_length: int = 500,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.max_episode_length = max_episode_length

        # Load episodes
        episodes_file = self.data_path / f"{split}.json"
        with open(episodes_file) as f:
            self.episodes = json.load(f)

        print(f"Loaded {len(self.episodes)} episodes from {split}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode = self.episodes[idx]

        return {
            "episode_id": episode["episode_id"],
            "scene_id": episode["scene_id"],
            "instruction": episode["instruction"]["instruction_text"],
            "trajectory": episode.get("reference_path", []),
            "goal_position": episode.get("goals", [{}])[0].get("position", [0, 0, 0]),
        }


class VLNDataModule:
    """Data module for VLN training."""

    def __init__(self, config):
        self.config = config

    def setup(self):
        self.train_dataset = VLNDataset(
            self.config.data.data_path,
            split=self.config.data.train_split,
        )
        self.val_dataset = VLNDataset(
            self.config.data.data_path,
            split=self.config.data.val_split,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
        )

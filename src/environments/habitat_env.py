"""
Habitat Environment Wrapper for VLA-GR Navigation
Provides complete integration with Habitat-Lab and Habitat-Sim
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

import habitat
from habitat import Config, Dataset
from habitat.core.env import Env
from habitat.tasks.nav.nav import NavigationTask, NavigationEpisode
from habitat.config.default import get_config
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class HabitatConfig:
    """Configuration for Habitat environment."""
    
    # Scene datasets
    scene_dataset: str = "hm3d"  # Options: hm3d, mp3d, gibson, replica
    split: str = "train"
    
    # Task configuration
    task: str = "objectnav"  # Options: pointnav, objectnav, vln, imagenav
    max_episode_steps: int = 500
    success_distance: float = 0.2  # meters
    
    # Sensor configuration
    rgb_sensor: bool = True
    depth_sensor: bool = True
    semantic_sensor: bool = True
    sensor_height: float = 0.88  # meters
    sensor_width: int = 640
    sensor_height_pixels: int = 480
    hfov: int = 79  # horizontal field of view in degrees
    
    # Agent configuration
    forward_step_size: float = 0.25  # meters
    turn_angle: int = 10  # degrees
    
    # Rewards
    success_reward: float = 10.0
    slack_penalty: float = -0.01
    collision_penalty: float = -0.1


class HabitatEnvironment:
    """
    Complete Habitat environment wrapper for VLA-GR navigation.
    
    Features:
    - Multiple scene dataset support (HM3D, MP3D, Gibson, Replica)
    - Various navigation tasks (PointNav, ObjectNav, VLN, ImageNav)
    - Comprehensive sensor suite (RGB-D, Semantic, GPS, Compass)
    - Action space: continuous and discrete
    - Reward shaping for navigation
    """
    
    def __init__(self, config: HabitatConfig):
        self.config = config
        
        # Create Habitat configuration
        self.habitat_config = self._create_habitat_config()
        
        # Initialize environment
        self.env = habitat.Env(config=self.habitat_config)
        
        # Episode dataset
        self.dataset = make_dataset(
            id_dataset=self.habitat_config.DATASET.TYPE,
            config=self.habitat_config.DATASET
        )
        
        # Action space
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Episode tracking
        self.current_episode = None
        self.episode_steps = 0
        self.episode_distance = 0.0
        self.trajectory = []
        
        # Metrics
        self.metrics = {
            'success': [],
            'spl': [],
            'distance_to_goal': [],
            'episode_length': [],
            'collisions': []
        }
        
        logger.info(f"Initialized Habitat environment with {config.scene_dataset} dataset")
        
    def _create_habitat_config(self) -> Config:
        """Create Habitat configuration based on task and dataset."""
        
        # Base configuration path based on task
        config_paths = {
            'pointnav': 'configs/tasks/pointnav.yaml',
            'objectnav': 'configs/tasks/objectnav_hm3d.yaml',
            'vln': 'configs/tasks/vln_r2r.yaml',
            'imagenav': 'configs/tasks/imagenav.yaml'
        }
        
        # Get base config
        config_path = config_paths.get(self.config.task, 'configs/tasks/pointnav.yaml')
        habitat_config = get_config(config_path)
        
        # Update configuration
        habitat_config.defrost()
        
        # Dataset configuration
        if self.config.scene_dataset == "hm3d":
            habitat_config.DATASET.TYPE = "ObjectNav-v1"
            habitat_config.DATASET.SCENES_DIR = "data/scene_datasets/hm3d"
            habitat_config.DATASET.DATA_PATH = "data/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz"
        elif self.config.scene_dataset == "mp3d":
            habitat_config.DATASET.TYPE = "PointNav-v1"
            habitat_config.DATASET.SCENES_DIR = "data/scene_datasets/mp3d"
            habitat_config.DATASET.DATA_PATH = "data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz"
        elif self.config.scene_dataset == "gibson":
            habitat_config.DATASET.TYPE = "PointNav-v1"
            habitat_config.DATASET.SCENES_DIR = "data/scene_datasets/gibson"
            habitat_config.DATASET.DATA_PATH = "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
        elif self.config.scene_dataset == "replica":
            habitat_config.DATASET.TYPE = "PointNav-v1"
            habitat_config.DATASET.SCENES_DIR = "data/scene_datasets/replica"
            habitat_config.DATASET.DATA_PATH = "data/datasets/pointnav/replica/v1/{split}/{split}.json.gz"
            
        habitat_config.DATASET.SPLIT = self.config.split
        
        # Simulator configuration
        habitat_config.SIMULATOR.TYPE = "Sim-v0"
        habitat_config.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
        habitat_config.SIMULATOR.FORWARD_STEP_SIZE = self.config.forward_step_size
        habitat_config.SIMULATOR.TURN_ANGLE = self.config.turn_angle
        
        # Sensor configuration
        habitat_config.SIMULATOR.AGENT_0.HEIGHT = self.config.sensor_height
        
        # RGB sensor
        if self.config.rgb_sensor:
            habitat_config.SIMULATOR.RGB_SENSOR.WIDTH = self.config.sensor_width
            habitat_config.SIMULATOR.RGB_SENSOR.HEIGHT = self.config.sensor_height_pixels
            habitat_config.SIMULATOR.RGB_SENSOR.HFOV = self.config.hfov
            habitat_config.SIMULATOR.RGB_SENSOR.POSITION = [0, self.config.sensor_height, 0]
            
        # Depth sensor
        if self.config.depth_sensor:
            habitat_config.SIMULATOR.DEPTH_SENSOR.WIDTH = self.config.sensor_width
            habitat_config.SIMULATOR.DEPTH_SENSOR.HEIGHT = self.config.sensor_height_pixels
            habitat_config.SIMULATOR.DEPTH_SENSOR.HFOV = self.config.hfov
            habitat_config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
            habitat_config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 10.0
            habitat_config.SIMULATOR.DEPTH_SENSOR.POSITION = [0, self.config.sensor_height, 0]
            
        # Semantic sensor
        if self.config.semantic_sensor:
            habitat_config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = self.config.sensor_width
            habitat_config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = self.config.sensor_height_pixels
            habitat_config.SIMULATOR.SEMANTIC_SENSOR.HFOV = self.config.hfov
            habitat_config.SIMULATOR.SEMANTIC_SENSOR.POSITION = [0, self.config.sensor_height, 0]
            
        # Task configuration
        habitat_config.TASK.TYPE = self.config.task
        habitat_config.TASK.SUCCESS_DISTANCE = self.config.success_distance
        habitat_config.TASK.MAX_EPISODE_STEPS = self.config.max_episode_steps
        
        # Measurements
        habitat_config.TASK.MEASUREMENTS = [
            "DISTANCE_TO_GOAL",
            "SUCCESS",
            "SPL",
            "COLLISIONS",
            "TOP_DOWN_MAP",
            "EPISODE_LENGTH"
        ]
        
        # Rewards
        habitat_config.TASK.SUCCESS_REWARD = self.config.success_reward
        habitat_config.TASK.SLACK_REWARD = self.config.slack_penalty
        habitat_config.TASK.COLLISION_REWARD = self.config.collision_penalty
        
        habitat_config.freeze()
        
        return habitat_config
    
    def reset(self, episode_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Reset environment to new episode.
        
        Args:
            episode_id: Specific episode to load, random if None
            
        Returns:
            Initial observation dictionary
        """
        
        # Select episode
        if episode_id is not None:
            episode = self._get_episode_by_id(episode_id)
        else:
            episode = self.dataset.get_episode(np.random.choice(len(self.dataset.episodes)))
            
        self.current_episode = episode
        
        # Reset environment with episode
        observations = self.env.reset()
        
        # Reset tracking
        self.episode_steps = 0
        self.episode_distance = 0.0
        self.trajectory = [self.env.sim.get_agent_state().position]
        
        # Process observations
        processed_obs = self._process_observations(observations)
        
        return processed_obs
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action in environment.
        
        Args:
            action: Action to execute (continuous or discrete)
            
        Returns:
            observation: Next observation
            reward: Step reward
            done: Episode terminated
            info: Additional information
        """
        
        # Convert continuous to discrete action if needed
        if isinstance(action, np.ndarray) and len(action) > 1:
            action = self._continuous_to_discrete(action)
            
        # Execute action
        observations = self.env.step(action)
        
        # Update tracking
        self.episode_steps += 1
        current_position = self.env.sim.get_agent_state().position
        self.trajectory.append(current_position)
        
        # Calculate reward
        reward = self._calculate_reward(observations)
        
        # Check termination
        done = self._is_episode_done(observations)
        
        # Get info
        info = self._get_step_info(observations)
        
        # Process observations
        processed_obs = self._process_observations(observations)
        
        return processed_obs, reward, done, info
    
    def _process_observations(self, observations: Dict) -> Dict[str, np.ndarray]:
        """Process raw observations from Habitat."""
        
        processed = {}
        
        # RGB image
        if "rgb" in observations:
            rgb = observations["rgb"]
            # Convert from [H, W, C] to [C, H, W] for PyTorch
            processed["rgb"] = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
            
        # Depth map
        if "depth" in observations:
            depth = observations["depth"]
            # Add channel dimension [H, W] -> [1, H, W]
            processed["depth"] = np.expand_dims(depth, axis=0).astype(np.float32)
            
        # Semantic segmentation
        if "semantic" in observations:
            semantic = observations["semantic"]
            processed["semantic"] = semantic.astype(np.int64)
            
        # GPS and Compass
        if "gps" in observations:
            processed["gps"] = observations["gps"].astype(np.float32)
            
        if "compass" in observations:
            processed["compass"] = observations["compass"].astype(np.float32)
            
        # Goal information
        if "pointgoal" in observations:
            processed["pointgoal"] = observations["pointgoal"].astype(np.float32)
            
        if "objectgoal" in observations:
            processed["objectgoal"] = observations["objectgoal"].astype(np.int64)
            
        # Add agent state
        agent_state = self.env.sim.get_agent_state()
        processed["position"] = np.array(agent_state.position, dtype=np.float32)
        processed["rotation"] = np.array(agent_state.rotation.components, dtype=np.float32)
        
        return processed
    
    def _continuous_to_discrete(self, action: np.ndarray) -> int:
        """Convert continuous action to discrete Habitat action."""
        
        # action[0:3] = velocity (forward/backward, left/right, up/down)
        # action[3:7] = rotation quaternion
        
        # Simple thresholding
        if action[0] > 0.5:  # Forward
            return HabitatSimActions.MOVE_FORWARD
        elif action[1] > 0.5:  # Turn right
            return HabitatSimActions.TURN_RIGHT
        elif action[1] < -0.5:  # Turn left
            return HabitatSimActions.TURN_LEFT
        else:
            return HabitatSimActions.STOP
            
    def _calculate_reward(self, observations: Dict) -> float:
        """Calculate step reward."""
        
        reward = 0.0
        
        # Success reward
        if self._check_success(observations):
            reward += self.config.success_reward
            
        # Distance progress reward
        if "distance_to_goal" in observations:
            prev_distance = self.prev_distance if hasattr(self, 'prev_distance') else observations["distance_to_goal"]
            distance_delta = prev_distance - observations["distance_to_goal"]
            reward += distance_delta
            self.prev_distance = observations["distance_to_goal"]
            
        # Slack penalty (time penalty)
        reward += self.config.slack_penalty
        
        # Collision penalty
        if self._check_collision(observations):
            reward += self.config.collision_penalty
            
        return reward
    
    def _is_episode_done(self, observations: Dict) -> bool:
        """Check if episode should terminate."""
        
        # Success
        if self._check_success(observations):
            return True
            
        # Max steps
        if self.episode_steps >= self.config.max_episode_steps:
            return True
            
        # Collision (optional termination)
        # if self._check_collision(observations):
        #     return True
            
        return False
    
    def _check_success(self, observations: Dict) -> bool:
        """Check if goal is reached."""
        
        if "success" in observations:
            return bool(observations["success"])
            
        if "distance_to_goal" in observations:
            return observations["distance_to_goal"] < self.config.success_distance
            
        return False
    
    def _check_collision(self, observations: Dict) -> bool:
        """Check if collision occurred."""
        
        if "collisions" in observations:
            if "is_collision" in observations["collisions"]:
                return bool(observations["collisions"]["is_collision"])
                
        # Check if agent moved less than expected
        if len(self.trajectory) > 1:
            expected_distance = self.config.forward_step_size
            actual_distance = np.linalg.norm(
                np.array(self.trajectory[-1]) - np.array(self.trajectory[-2])
            )
            if actual_distance < expected_distance * 0.5:
                return True
                
        return False
    
    def _get_step_info(self, observations: Dict) -> Dict:
        """Get additional step information."""
        
        info = {
            "episode_steps": self.episode_steps,
            "distance_to_goal": observations.get("distance_to_goal", -1),
            "success": self._check_success(observations),
            "collision": self._check_collision(observations),
            "episode_id": self.current_episode.episode_id if self.current_episode else None,
            "scene_id": self.current_episode.scene_id if self.current_episode else None
        }
        
        # Add metrics
        if "spl" in observations:
            info["spl"] = observations["spl"]
            
        return info
    
    def _get_episode_by_id(self, episode_id: str) -> NavigationEpisode:
        """Get specific episode by ID."""
        
        for episode in self.dataset.episodes:
            if episode.episode_id == episode_id:
                return episode
                
        raise ValueError(f"Episode {episode_id} not found")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics."""
        
        metrics = {}
        
        for key, values in self.metrics.items():
            if values:
                metrics[f"mean_{key}"] = np.mean(values)
                metrics[f"std_{key}"] = np.std(values)
                
        return metrics
    
    def render(self, mode: str = "rgb") -> np.ndarray:
        """Render current observation."""
        
        observations = self.env.sim.get_sensor_observations()
        
        if mode == "rgb":
            return observations.get("rgb", np.zeros((480, 640, 3)))
        elif mode == "depth":
            return observations.get("depth", np.zeros((480, 640)))
        elif mode == "semantic":
            return observations.get("semantic", np.zeros((480, 640)))
        elif mode == "topdown":
            return self._get_topdown_map()
        else:
            raise ValueError(f"Unknown render mode: {mode}")
            
    def _get_topdown_map(self) -> np.ndarray:
        """Get top-down map with trajectory."""
        
        top_down_map = maps.get_topdown_map_from_sim(
            self.env.sim,
            map_resolution=1024,
            draw_border=True
        )
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            trajectory_points = np.array(self.trajectory)
            # Convert world coordinates to map coordinates
            # This is simplified - actual implementation would need proper transformation
            
        return top_down_map
    
    def close(self):
        """Close environment."""
        
        if hasattr(self, 'env'):
            self.env.close()


class HabitatDatasetManager:
    """
    Manage Habitat datasets and episode generation.
    
    Supports:
    - HM3D (Habitat-Matterport 3D)
    - MP3D (Matterport3D)
    - Gibson
    - Replica
    """
    
    def __init__(self, dataset_type: str = "hm3d"):
        self.dataset_type = dataset_type
        self.dataset_path = self._get_dataset_path()
        
    def _get_dataset_path(self) -> str:
        """Get dataset path based on type."""
        
        paths = {
            "hm3d": "data/scene_datasets/hm3d",
            "mp3d": "data/scene_datasets/mp3d",
            "gibson": "data/scene_datasets/gibson",
            "replica": "data/scene_datasets/replica"
        }
        
        return paths.get(self.dataset_type, "data/scene_datasets/hm3d")
    
    def download_dataset(self):
        """Download dataset if not present."""
        
        import subprocess
        
        if self.dataset_type == "hm3d":
            # Download HM3D dataset
            cmd = [
                "python", "-m", "habitat_sim.utils.datasets_download",
                "--uids", "hm3d_minival",
                "--data-path", "data/"
            ]
            subprocess.run(cmd)
            
        elif self.dataset_type == "replica":
            # Download Replica dataset
            cmd = [
                "python", "-m", "habitat_sim.utils.datasets_download",
                "--uids", "replica_cad_dataset",
                "--data-path", "data/"
            ]
            subprocess.run(cmd)
            
        logger.info(f"Downloaded {self.dataset_type} dataset")
    
    def generate_episodes(
        self,
        num_episodes: int,
        split: str = "train",
        task: str = "pointnav"
    ) -> List[Dict]:
        """Generate navigation episodes."""
        
        episodes = []
        
        # Load scene list
        scenes = self._get_scene_list(split)
        
        for i in range(num_episodes):
            scene = np.random.choice(scenes)
            
            if task == "pointnav":
                episode = self._generate_pointnav_episode(scene, i)
            elif task == "objectnav":
                episode = self._generate_objectnav_episode(scene, i)
            else:
                raise ValueError(f"Unknown task: {task}")
                
            episodes.append(episode)
            
        return episodes
    
    def _get_scene_list(self, split: str) -> List[str]:
        """Get list of available scenes."""
        
        # This would load actual scene lists from dataset
        # For now, returning example scenes
        
        if self.dataset_type == "hm3d":
            if split == "train":
                return [f"hm3d_train_scene_{i}" for i in range(100)]
            else:
                return [f"hm3d_val_scene_{i}" for i in range(20)]
                
        elif self.dataset_type == "replica":
            return ["apartment_0", "apartment_1", "apartment_2", "office_0", "office_1"]
            
        return []
    
    def _generate_pointnav_episode(self, scene: str, episode_id: int) -> Dict:
        """Generate PointNav episode."""
        
        return {
            "episode_id": f"pointnav_{episode_id}",
            "scene_id": scene,
            "start_position": [0, 0, 0],  # Would be sampled from navigable points
            "start_rotation": [0, 0, 0, 1],
            "goals": [{
                "position": [5, 0, 5],  # Would be sampled
                "radius": 0.2
            }]
        }
    
    def _generate_objectnav_episode(self, scene: str, episode_id: int) -> Dict:
        """Generate ObjectNav episode."""
        
        object_categories = ["chair", "table", "bed", "toilet", "couch", "plant"]
        
        return {
            "episode_id": f"objectnav_{episode_id}",
            "scene_id": scene,
            "start_position": [0, 0, 0],
            "start_rotation": [0, 0, 0, 1],
            "goals": [{
                "object_category": np.random.choice(object_categories),
                "object_id": f"object_{np.random.randint(100)}"
            }]
        }

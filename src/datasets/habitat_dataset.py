"""
Habitat dataset for VLA-GR navigation training.
Provides RGB-D observations, language instructions, and navigation targets.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

import habitat
from habitat.config import Config
from habitat.core.simulator import Simulator
from habitat.config.default import get_config
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.utils.visualizations import maps
from habitat.sims import make_sim

# Import make_dataset for proper dataset loading (Habitat 0.3.3+)
try:
    from habitat.datasets import make_dataset
    HAS_MAKE_DATASET = True
except ImportError:
    logger.warning("make_dataset not available, using legacy dataset loading")
    HAS_MAKE_DATASET = False

import logging

logger = logging.getLogger(__name__)


class HabitatNavigationDataset(Dataset):
    """
    Dataset for navigation tasks in Habitat simulator.
    
    Features:
    - RGB-D observations from simulated environments
    - Natural language navigation instructions
    - Ground truth paths and actions
    - Dynamic scene generation
    """
    
    def __init__(
        self,
        config: Dict,
        split: str = "train",
        transform: Optional[Any] = None,
        precompute_features: bool = False
    ):
        super().__init__()
        self.config = config
        self.split = split
        self.transform = transform
        self.precompute_features = precompute_features
        
        # Setup Habitat
        self.habitat_config = self._create_habitat_config()
        # Use make_sim to create simulator properly
        try:
            self.simulator = make_sim(
                id_sim=self.habitat_config.SIMULATOR.TYPE,
                config=self.habitat_config.SIMULATOR
            )
        except Exception as e:
            logger.warning(f"Failed to create simulator with make_sim: {e}")
            # Fallback to direct instantiation
            self.simulator = Simulator(self.habitat_config)
        
        # Load episodes
        self.episodes = self._load_episodes()
        logger.info(f"Loaded {len(self.episodes)} episodes for {split}")
        
        # Language instruction templates
        self.instruction_templates = self._load_instruction_templates()
        
        # Object categories for goal specification
        self.object_categories = config['environment']['task']['goals']
        
        # Precomputed features cache
        self.feature_cache = {} if precompute_features else None
        
        # Data augmentation
        self.augment_config = config['training']['augment'] if split == "train" else None
        
    def _create_habitat_config(self) -> Config:
        """Create Habitat simulator configuration."""
        # Use get_config from habitat.config.default
        try:
            config = get_config(config_paths="configs/tasks/pointnav_gibson.yaml")
        except (FileNotFoundError, RuntimeError, Exception) as e:
            # Fallback: create basic config
            logger.warning(f"Could not load config file: {e}")
            config = get_config()
        
        # Update with our config
        config.defrost()
        
        # Simulator settings
        config.SIMULATOR.SCENE_DATASET = self.config['environment']['habitat']['scene_dataset']
        config.SIMULATOR.TURN_ANGLE = 10
        config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
        
        # Sensor configurations
        # RGB sensor
        config.SIMULATOR.RGB_SENSOR.WIDTH = self.config['environment']['sensors']['rgb']['width']
        config.SIMULATOR.RGB_SENSOR.HEIGHT = self.config['environment']['sensors']['rgb']['height']
        config.SIMULATOR.RGB_SENSOR.HFOV = self.config['environment']['sensors']['rgb']['fov']
        
        # Depth sensor
        config.SIMULATOR.DEPTH_SENSOR.WIDTH = self.config['environment']['sensors']['depth']['width']
        config.SIMULATOR.DEPTH_SENSOR.HEIGHT = self.config['environment']['sensors']['depth']['height']
        config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = self.config['environment']['sensors']['depth']['min_depth']
        config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = self.config['environment']['sensors']['depth']['max_depth']
        
        # Semantic sensor
        config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = self.config['environment']['sensors']['semantic']['width']
        config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = self.config['environment']['sensors']['semantic']['height']
        
        config.freeze()
        
        return config
    
    def _load_episodes(self) -> List[NavigationEpisode]:
        """Load navigation episodes."""
        # Load from file or generate
        episodes_file = Path(f"data/episodes_{self.split}.json")
        
        if episodes_file.exists():
            with open(episodes_file, 'r') as f:
                episodes_data = json.load(f)
            episodes = [self._dict_to_episode(e) for e in episodes_data]
        else:
            # Generate episodes
            episodes = self._generate_episodes(
                num_episodes=1000 if self.split == "train" else 100
            )
            # Save for reproducibility
            self._save_episodes(episodes, episodes_file)
            
        return episodes
    
    def _generate_episodes(self, num_episodes: int) -> List[NavigationEpisode]:
        """Generate random navigation episodes."""
        episodes = []

        for i in range(num_episodes):
            # Get scene ID from the current loaded scene
            # semantic_scene.levels[0].id is a string, not a list
            try:
                if hasattr(self.simulator, 'semantic_scene') and self.simulator.semantic_scene:
                    scene_id = self.simulator.semantic_scene.levels[0].id
                else:
                    # Fallback to default scene ID
                    scene_id = f"scene_{i % 10}"
            except (AttributeError, IndexError):
                scene_id = f"scene_{i % 10}"

            # Random start position
            start_position = self.simulator.sample_navigable_point()
            start_rotation = [0, random.uniform(0, 2 * np.pi), 0, 1]

            # Random goal
            goal_position = self.simulator.sample_navigable_point()
            goal_radius = self.config['environment']['habitat']['success_distance']

            # Create episode
            episode = NavigationEpisode(
                episode_id=f"{self.split}_{i}",
                scene_id=scene_id,
                start_position=start_position,
                start_rotation=start_rotation,
                goals=[NavigationGoal(position=goal_position, radius=goal_radius)]
            )

            episodes.append(episode)

        return episodes
    
    def _save_episodes(self, episodes: List[NavigationEpisode], file_path: Path):
        """Save episodes to file."""
        episodes_data = [self._episode_to_dict(e) for e in episodes]
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(episodes_data, f, indent=2)
            
    def _episode_to_dict(self, episode: NavigationEpisode) -> Dict:
        """Convert episode to dictionary."""
        return {
            'episode_id': episode.episode_id,
            'scene_id': episode.scene_id,
            'start_position': episode.start_position,
            'start_rotation': episode.start_rotation,
            'goals': [
                {
                    'position': g.position,
                    'radius': g.radius
                }
                for g in episode.goals
            ]
        }
    
    def _dict_to_episode(self, data: Dict) -> NavigationEpisode:
        """Convert dictionary to episode."""
        return NavigationEpisode(
            episode_id=data['episode_id'],
            scene_id=data['scene_id'],
            start_position=data['start_position'],
            start_rotation=data['start_rotation'],
            goals=[
                NavigationGoal(
                    position=g['position'],
                    radius=g['radius']
                )
                for g in data['goals']
            ]
        )
    
    def _load_instruction_templates(self) -> List[str]:
        """Load language instruction templates."""
        templates = [
            "Navigate to the {object}",
            "Go to the {object} in the room",
            "Find the {object}",
            "Move towards the {object}",
            "Walk to the {object}",
            "Reach the {object}",
            "Head over to the {object}",
            "Make your way to the {object}",
            "Travel to the {object}",
            "Proceed to the {object}"
        ]
        
        # Add more complex templates
        complex_templates = [
            "Navigate through the room and reach the {object}",
            "Find and go to the {object} while avoiding obstacles",
            "Carefully make your way to the {object}",
            "Use the safest path to reach the {object}",
            "Navigate around furniture to get to the {object}"
        ]
        
        return templates + complex_templates
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training sample."""
        
        episode = self.episodes[idx]
        
        # Reset simulator to episode
        self.simulator.reset()
        self.simulator.set_agent_state(
            position=episode.start_position,
            rotation=episode.start_rotation
        )
        
        # Get observations
        obs = self.simulator.get_sensor_observations()
        
        # Process RGB
        rgb = obs['rgb']
        if self.transform:
            rgb = self.transform(rgb)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        
        # Process depth
        depth = obs['depth']
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        
        # Process semantic (if available)
        semantic = None
        if 'semantic' in obs:
            semantic = torch.from_numpy(obs['semantic']).long()
            
        # Generate language instruction
        object_goal = random.choice(self.object_categories)
        template = random.choice(self.instruction_templates)
        instruction = template.format(object=object_goal)
        
        # Generate optimal path (using A* or shortest path)
        optimal_path = self._compute_optimal_path(
            start=episode.start_position,
            goal=episode.goals[0].position
        )
        
        # Compute target actions
        target_actions = self._path_to_actions(optimal_path)
        
        # Generate affordance map (ground truth)
        affordance_map = self._generate_affordance_map(obs, semantic)
        
        # Add noise/occlusions if training
        if self.split == "train" and self.augment_config:
            rgb, depth = self._apply_augmentations(rgb, depth)
            
        # Prepare sample
        sample = {
            'rgb': rgb,
            'depth': depth,
            'instruction': instruction,
            'episode_id': episode.episode_id,
            'position': torch.tensor(episode.start_position, dtype=torch.float32),
            'orientation': torch.tensor(episode.start_rotation, dtype=torch.float32),
            'velocity': torch.zeros(3, dtype=torch.float32),
            'targets': {
                'target_actions': torch.tensor(target_actions, dtype=torch.float32),
                'optimal_path': torch.tensor(optimal_path, dtype=torch.float32),
                'target_affordance': affordance_map,
                'target_depth': depth.clone(),
                'goal_position': torch.tensor(episode.goals[0].position, dtype=torch.float32)
            }
        }
        
        # Add semantic if available
        if semantic is not None:
            sample['semantic'] = semantic
            sample['targets']['semantic_labels'] = semantic
            
        return sample
    
    def _compute_optimal_path(
        self,
        start: List[float],
        goal: List[float]
    ) -> np.ndarray:
        """Compute optimal path from start to goal."""
        
        # Get navigability map
        top_down_map = maps.get_topdown_map_from_sim(
            self.simulator,
            map_resolution=1024
        )
        
        # Convert positions to map coordinates
        start_map = self._world_to_map(start, top_down_map.shape)
        goal_map = self._world_to_map(goal, top_down_map.shape)
        
        # Use A* pathfinding
        path = self._astar(top_down_map, start_map, goal_map)
        
        # Convert back to world coordinates
        world_path = np.array([
            self._map_to_world(p, top_down_map.shape)
            for p in path
        ])
        
        return world_path
    
    def _astar(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm."""
        
        import heapq
        
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        
        # Initialize
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            # Check neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if (0 <= neighbor[0] < grid.shape[0] and
                    0 <= neighbor[1] < grid.shape[1]):
                    
                    # Check navigability
                    if grid[neighbor[0], neighbor[1]] == 0:
                        continue
                        
                    tentative_g = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
        # No path found
        return [(start[0], start[1]), (goal[0], goal[1])]
    
    def _world_to_map(
        self,
        world_pos: List[float],
        map_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Convert world coordinates to map coordinates."""
        # Simplified - would need actual transformation
        x = int((world_pos[0] + 10) / 20 * map_shape[0])
        y = int((world_pos[2] + 10) / 20 * map_shape[1])
        return (x, y)
    
    def _map_to_world(
        self,
        map_pos: Tuple[int, int],
        map_shape: Tuple[int, int]
    ) -> List[float]:
        """Convert map coordinates to world coordinates."""
        x = map_pos[0] / map_shape[0] * 20 - 10
        z = map_pos[1] / map_shape[1] * 20 - 10
        return [x, 0.0, z]
    
    def _path_to_actions(self, path: np.ndarray) -> np.ndarray:
        """Convert path to action sequence."""
        
        if len(path) < 2:
            return np.zeros((1, 7))  # Default action
            
        actions = []
        
        for i in range(len(path) - 1):
            # Compute direction
            direction = path[i + 1] - path[i]
            
            # Normalize
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Create action (velocity + zero rotation)
            action = np.concatenate([direction, np.zeros(4)])
            actions.append(action)
            
        return np.array(actions)
    
    def _generate_affordance_map(
        self,
        obs: Dict,
        semantic: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Generate ground truth affordance map."""
        
        H, W = obs['depth'].shape[:2]
        affordance = torch.zeros(H, W, len(self.object_categories))
        
        if semantic is not None:
            # Map semantic labels to affordance values
            for i, category in enumerate(self.object_categories):
                # Simplified - would use actual semantic mapping
                mask = (semantic == i)
                affordance[..., i] = mask.float()
                
        else:
            # Use depth-based heuristics
            depth = obs['depth']
            
            # Close objects have higher affordance
            close_mask = (depth > 0) & (depth < 2.0)
            affordance[..., 0] = torch.from_numpy(close_mask).float()
            
            # Far objects
            far_mask = (depth >= 2.0) & (depth < 5.0)
            affordance[..., 1] = torch.from_numpy(far_mask).float()
            
        return affordance
    
    def _apply_augmentations(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentations."""
        
        # Random occlusions in depth
        if self.augment_config.get('depth_noise', 0) > 0:
            mask = torch.rand_like(depth) < self.augment_config['depth_noise']
            depth = depth * ~mask  # Set occluded pixels to 0
            
        # Gaussian noise in RGB
        if self.augment_config.get('gaussian_noise', 0) > 0:
            noise = torch.randn_like(rgb) * self.augment_config['gaussian_noise']
            rgb = torch.clamp(rgb + noise, 0, 1)
            
        return rgb, depth
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function for batching."""
        
        # Stack tensors
        collated = {}
        
        # Simple fields
        for key in ['rgb', 'depth', 'position', 'orientation', 'velocity']:
            if key in batch[0]:
                collated[key] = torch.stack([b[key] for b in batch])
                
        # Lists
        collated['instruction'] = [b['instruction'] for b in batch]
        collated['episode_id'] = [b['episode_id'] for b in batch]
        
        # Nested targets
        collated['targets'] = {}
        for key in batch[0]['targets']:
            if isinstance(batch[0]['targets'][key], torch.Tensor):
                collated['targets'][key] = torch.stack([
                    b['targets'][key] for b in batch
                ])
                
        return collated

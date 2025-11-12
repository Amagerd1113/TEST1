"""
Advanced Habitat 0.3.3 Environment Wrapper for VLA-GR Navigation
State-of-the-art integration with Habitat-Lab 0.3.3 for conference-level research
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from collections import defaultdict
try:
    import gymnasium as gym
except ImportError:
    import gym  # Fallback to gym for habitat 0.3.3
from enum import Enum

import habitat
from habitat import Config, Env
from habitat.core.env import Env, RLEnv
from habitat.tasks.nav.nav import NavigationTask, NavigationGoal
try:
    from habitat.tasks.nav.object_nav_task import ObjectGoal
except ImportError:
    ObjectGoal = None
try:
    from habitat.tasks.utils import cartesian_to_polar
except ImportError:
    def cartesian_to_polar(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return rho, phi

# Try to import structured configs (habitat 0.3.x), fallback to legacy
try:
    from habitat.config.default_structured_configs import (
        HabitatConfigPlugin,
        SimulatorConfig,
        HabitatSimV0Config,
    )
    HAS_STRUCTURED_CONFIGS = True
except ImportError:
    HAS_STRUCTURED_CONFIGS = False

from habitat.utils.visualizations import maps

# Geometry utils with fallback
try:
    from habitat.utils.geometry_utils import quaternion_to_list, quaternion_from_coeff
except ImportError:
    import quaternion as npq
    def quaternion_to_list(q):
        if hasattr(q, 'components'):
            return q.components.tolist()
        return [q.x, q.y, q.z, q.w]
    def quaternion_from_coeff(coeffs):
        return npq.quaternion(*coeffs)

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import cv2

# Habitat 0.3.3 specific imports with fallbacks
try:
    from habitat.gym import make_gym_from_config
except ImportError:
    make_gym_from_config = None

try:
    from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoal
except ImportError:
    InstanceImageGoal = None

try:
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
except ImportError:
    ShortestPathFollower = None

try:
    from habitat_baselines.common.obs_transformers import (
        apply_obs_transforms_batch,
        apply_obs_transforms_obs_space,
        get_active_obs_transforms,
    )
    HAS_OBS_TRANSFORMS = True
except ImportError:
    HAS_OBS_TRANSFORMS = False
    # Define fallback functions
    def apply_obs_transforms_batch(obs, transforms):
        return obs
    def apply_obs_transforms_obs_space(space, transforms):
        return space
    def get_active_obs_transforms(config):
        return []

logger = logging.getLogger(__name__)


@dataclass
class AdvancedHabitatConfig:
    """Enhanced configuration for Habitat 0.3.3 with research-grade features."""
    
    # Scene configuration
    scene_dataset: str = "hm3d-v0.2"  # Latest HM3D version
    scene_split: str = "train"
    additional_scenes_dir: Optional[str] = None
    
    # Task configuration
    task_type: str = "ObjectNav-v2"  # Updated task version
    max_episode_steps: int = 500
    success_distance: float = 0.1  # Stricter success criteria
    geodesic_distance_limit: float = 30.0
    
    # Advanced sensor suite
    sensors: Dict[str, bool] = field(default_factory=lambda: {
        "rgb": True,
        "depth": True,
        "semantic": True,
        "instance": True,
        "panoptic": True,
        "normal": True,
        "equirect": False,
        "fisheye": False
    })
    
    # Sensor specifications
    sensor_resolution: Tuple[int, int] = (480, 640)  # H, W
    hfov: float = 79.0
    sensor_position: List[float] = field(default_factory=lambda: [0.0, 0.88, 0.0])
    sensor_orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # Agent dynamics
    agent_dynamics: str = "kinematic"  # Options: kinematic, dynamic
    forward_step_size: float = 0.25
    turn_angle: float = 10.0
    tilt_angle: float = 15.0  # For looking up/down
    max_velocity: float = 1.0  # m/s
    max_angular_velocity: float = 90.0  # deg/s
    
    # Advanced features
    continuous_actions: bool = True
    allow_sliding: bool = True
    enable_physics: bool = True
    realistic_sensor_noise: bool = True
    
    # Reward shaping
    reward_config: Dict[str, float] = field(default_factory=lambda: {
        "success": 10.0,
        "progress": 0.1,
        "collision": -0.5,
        "slack": -0.01,
        "exploration": 0.05,
        "smoothness": 0.02,
        "field_following": 0.1  # Novel: reward for following GR field
    })
    
    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {"max_distance": 5.0, "obstacles": "low", "episodes": 10000},
        {"max_distance": 10.0, "obstacles": "medium", "episodes": 20000},
        {"max_distance": 15.0, "obstacles": "high", "episodes": 30000},
        {"max_distance": 30.0, "obstacles": "very_high", "episodes": -1}
    ])


class HabitatEnvV3(RLEnv):
    """
    State-of-the-art Habitat 0.3.3 environment wrapper for VLA-GR.
    
    Key Features:
    - Multi-modal sensor fusion with noise modeling
    - Continuous and discrete action spaces
    - Advanced reward shaping with field-based guidance
    - Curriculum learning support
    - Comprehensive metrics tracking
    - Real-time visualization capabilities
    """
    
    def __init__(
        self,
        config: Union[DictConfig, AdvancedHabitatConfig],
        dataset: Optional[str] = None,
        render_mode: Optional[str] = None
    ):
        """Initialize advanced Habitat environment."""
        
        if isinstance(config, AdvancedHabitatConfig):
            self.adv_config = config
            habitat_config = self._create_habitat_config()
        else:
            habitat_config = config
            self.adv_config = AdvancedHabitatConfig()
            
        # Initialize base environment
        super().__init__(habitat_config, dataset)
        
        # Advanced components
        if ShortestPathFollower is not None:
            try:
                self.shortest_path_follower = ShortestPathFollower(
                    self._env.sim,
                    goal_radius=self.adv_config.success_distance,
                    return_one_hot=False
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ShortestPathFollower: {e}")
                self.shortest_path_follower = None
        else:
            self.shortest_path_follower = None
        
        # Observation transformers for Habitat 0.3.3
        self.obs_transforms = get_active_obs_transforms(habitat_config)
        self.observation_space = apply_obs_transforms_obs_space(
            self.observation_space,
            self.obs_transforms
        )
        
        # Action space configuration
        if self.adv_config.continuous_actions:
            self.action_space = self._create_continuous_action_space()
        else:
            self.action_space = self._env.action_space
            
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.episode_metrics = defaultdict(list)
        
        # Curriculum learning
        self.curriculum_stage = 0
        self.total_episodes = 0
        
        # Visualization
        self.render_mode = render_mode
        self.video_writer = None
        
        # GR field cache for reward computation
        self.current_gr_field = None
        self.field_guidance_weight = 0.1
        
        logger.info(f"Initialized HabitatEnvV3 with Habitat {habitat.__version__}")
        
    def _create_habitat_config(self) -> Config:
        """Create Habitat 0.3.3 configuration."""
        
        # Use structured configs for Habitat 0.3.3
        from hydra import compose, initialize_config_dir
        from habitat.config.default_structured_configs import register_hydra_plugins
        
        register_hydra_plugins()
        
        config_dict = {
            "habitat": {
                "dataset": {
                    "type": self.adv_config.task_type,
                    "split": self.adv_config.scene_split,
                    "scenes_dir": f"data/scene_datasets/{self.adv_config.scene_dataset}",
                    "data_path": f"data/datasets/{self.adv_config.task_type}/{self.adv_config.scene_dataset}/{{split}}/{{split}}.json.gz"
                },
                "task": {
                    "type": self.adv_config.task_type,
                    "max_episode_steps": self.adv_config.max_episode_steps,
                    "success_distance": self.adv_config.success_distance,
                    "measurements": [
                        "distance_to_goal",
                        "success",
                        "spl",
                        "soft_spl",
                        "path_length",
                        "oracle_navigation_error",
                        "steps_taken",
                        "collisions"
                    ],
                    "lab_sensors": self._get_sensor_configs(),
                    "actions": self._get_action_configs()
                },
                "simulator": {
                    "type": "Sim-v0",
                    "scene_dataset": self.adv_config.scene_dataset,
                    "agents": {
                        "main_agent": {
                            "height": self.adv_config.sensor_position[1],
                            "radius": 0.18,
                            "sim_sensors": self._get_sim_sensor_configs()
                        }
                    },
                    "habitat_sim_v0": {
                        "gpu_device_id": 0,
                        "allow_sliding": self.adv_config.allow_sliding,
                        "enable_physics": self.adv_config.enable_physics
                    }
                },
                "environment": {
                    "max_episode_steps": self.adv_config.max_episode_steps
                }
            }
        }
        
        # Convert to OmegaConf
        config = OmegaConf.create(config_dict)
        
        # Merge with defaults
        from habitat.config.default import get_config as get_default_config
        default_config = get_default_config()
        config = OmegaConf.merge(default_config, config)
        
        return config
    
    def _get_sensor_configs(self) -> List[Dict]:
        """Get sensor configurations for Habitat 0.3.3."""
        
        sensors = []
        
        # RGB sensor
        if self.adv_config.sensors["rgb"]:
            sensors.append({
                "type": "RGB_SENSOR",
                "width": self.adv_config.sensor_resolution[1],
                "height": self.adv_config.sensor_resolution[0],
                "hfov": self.adv_config.hfov,
                "position": self.adv_config.sensor_position,
                "orientation": self.adv_config.sensor_orientation
            })
            
        # Depth sensor
        if self.adv_config.sensors["depth"]:
            sensors.append({
                "type": "DEPTH_SENSOR",
                "width": self.adv_config.sensor_resolution[1],
                "height": self.adv_config.sensor_resolution[0],
                "hfov": self.adv_config.hfov,
                "min_depth": 0.0,
                "max_depth": 10.0,
                "position": self.adv_config.sensor_position,
                "orientation": self.adv_config.sensor_orientation
            })
            
        # Semantic sensor
        if self.adv_config.sensors["semantic"]:
            sensors.append({
                "type": "SEMANTIC_SENSOR",
                "width": self.adv_config.sensor_resolution[1],
                "height": self.adv_config.sensor_resolution[0],
                "hfov": self.adv_config.hfov,
                "position": self.adv_config.sensor_position,
                "orientation": self.adv_config.sensor_orientation
            })
            
        # Instance segmentation
        if self.adv_config.sensors["instance"]:
            sensors.append({
                "type": "INSTANCE_SENSOR",
                "width": self.adv_config.sensor_resolution[1],
                "height": self.adv_config.sensor_resolution[0]
            })
            
        # Surface normals
        if self.adv_config.sensors["normal"]:
            sensors.append({
                "type": "NORMAL_SENSOR",
                "width": self.adv_config.sensor_resolution[1],
                "height": self.adv_config.sensor_resolution[0]
            })
            
        return sensors
    
    def _get_sim_sensor_configs(self) -> Dict:
        """Get simulator sensor configurations."""
        
        sim_sensors = {}
        
        for sensor_name, enabled in self.adv_config.sensors.items():
            if enabled:
                sensor_key = f"{sensor_name.upper()}_SENSOR"
                sim_sensors[sensor_key] = {
                    "width": self.adv_config.sensor_resolution[1],
                    "height": self.adv_config.sensor_resolution[0],
                    "hfov": self.adv_config.hfov,
                    "position": self.adv_config.sensor_position,
                    "orientation": self.adv_config.sensor_orientation
                }
                
        return sim_sensors
    
    def _get_action_configs(self) -> Dict:
        """Get action space configuration."""
        
        if self.adv_config.continuous_actions:
            return {
                "move_forward": {
                    "type": "MoveForwardAction",
                    "min": 0.0,
                    "max": self.adv_config.forward_step_size
                },
                "turn": {
                    "type": "TurnAction",
                    "min": -self.adv_config.turn_angle,
                    "max": self.adv_config.turn_angle
                },
                "look": {
                    "type": "LookAction",
                    "min": -self.adv_config.tilt_angle,
                    "max": self.adv_config.tilt_angle
                }
            }
        else:
            return {
                "move_forward": {"type": "MoveForwardAction"},
                "turn_left": {"type": "TurnLeftAction"},
                "turn_right": {"type": "TurnRightAction"},
                "stop": {"type": "StopAction"}
            }
    
    def _create_continuous_action_space(self) -> gym.Space:
        """Create continuous action space."""
        
        # [forward_velocity, angular_velocity, tilt_velocity, grasp, push]
        low = np.array([
            -self.adv_config.max_velocity,
            -self.adv_config.max_angular_velocity,
            -self.adv_config.tilt_angle,
            0.0,  # Grasp action
            0.0   # Push action
        ])
        
        high = np.array([
            self.adv_config.max_velocity,
            self.adv_config.max_angular_velocity,
            self.adv_config.tilt_angle,
            1.0,  # Grasp action
            1.0   # Push action
        ])
        
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset environment with Habitat 0.3.3 API."""
        
        # Update curriculum if enabled
        if self.adv_config.curriculum_enabled:
            self._update_curriculum()
            
        # Reset base environment
        observations = super().reset(seed=seed)
        
        # Apply observation transforms
        observations = apply_obs_transforms_batch(
            observations,
            self.obs_transforms
        )
        
        # Initialize episode tracking
        self.episode_metrics.clear()
        self.metrics_tracker.reset_episode()
        
        # Add advanced observations
        observations = self._add_advanced_observations(observations)
        
        # Apply sensor noise if enabled
        if self.adv_config.realistic_sensor_noise:
            observations = self._apply_sensor_noise(observations)
            
        info = self._get_info(observations)
        
        return observations, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action with advanced reward computation."""
        
        # Store previous state for reward computation
        prev_position = self._env.sim.get_agent_state().position
        prev_distance_to_goal = self._env.get_metrics()["distance_to_goal"]
        
        # Convert continuous to discrete if needed
        if self.adv_config.continuous_actions:
            habitat_action = self._continuous_to_discrete_action(action)
        else:
            habitat_action = action
            
        # Execute action in base environment
        observations = self._env.step(habitat_action)
        
        # Apply observation transforms
        observations = apply_obs_transforms_batch(
            observations,
            self.obs_transforms
        )
        
        # Add advanced observations
        observations = self._add_advanced_observations(observations)
        
        # Apply sensor noise
        if self.adv_config.realistic_sensor_noise:
            observations = self._apply_sensor_noise(observations)
            
        # Compute advanced reward
        reward = self._compute_advanced_reward(
            observations,
            action,
            prev_position,
            prev_distance_to_goal
        )
        
        # Get termination signals
        terminated = self._env.episode_over
        truncated = self._env.get_metrics()["steps_taken"] >= self.adv_config.max_episode_steps
        
        # Update metrics
        self.metrics_tracker.update(observations, reward, terminated)
        
        # Get info
        info = self._get_info(observations)
        info["curriculum_stage"] = self.curriculum_stage
        
        # Update episode count on termination
        if terminated or truncated:
            self.total_episodes += 1
            info["episode_metrics"] = self.metrics_tracker.get_episode_summary()
            
        return observations, reward, terminated, truncated, info
    
    def _continuous_to_discrete_action(self, action: np.ndarray) -> int:
        """Convert continuous action to discrete Habitat action."""
        
        # Extract components
        forward_vel = action[0]
        angular_vel = action[1]
        
        # Threshold-based conversion
        if abs(forward_vel) > abs(angular_vel):
            if forward_vel > 0.1:
                return 0  # MOVE_FORWARD
            else:
                return 3  # STOP
        else:
            if angular_vel > 0.1:
                return 2  # TURN_RIGHT
            elif angular_vel < -0.1:
                return 1  # TURN_LEFT
            else:
                return 3  # STOP
                
    def _compute_advanced_reward(
        self,
        observations: Dict,
        action: np.ndarray,
        prev_position: np.ndarray,
        prev_distance_to_goal: float
    ) -> float:
        """Compute sophisticated reward with multiple components."""
        
        reward = 0.0
        metrics = self._env.get_metrics()
        reward_config = self.adv_config.reward_config
        
        # Success reward
        if metrics["success"]:
            reward += reward_config["success"]
            
        # Progress reward (distance to goal reduction)
        distance_delta = prev_distance_to_goal - metrics["distance_to_goal"]
        reward += reward_config["progress"] * distance_delta
        
        # Collision penalty
        if metrics.get("collisions", {}).get("is_collision", False):
            reward += reward_config["collision"]
            
        # Time penalty (slack)
        reward += reward_config["slack"]
        
        # Exploration bonus (distance traveled)
        current_position = self._env.sim.get_agent_state().position
        distance_traveled = np.linalg.norm(current_position - prev_position)
        reward += reward_config["exploration"] * distance_traveled
        
        # Action smoothness reward (penalize jerky movements)
        if hasattr(self, "prev_action"):
            action_delta = np.linalg.norm(action - self.prev_action)
            reward -= reward_config["smoothness"] * action_delta
        self.prev_action = action.copy()
        
        # GR field following reward (novel contribution)
        if self.current_gr_field is not None:
            field_alignment = self._compute_field_alignment(
                current_position,
                action,
                self.current_gr_field
            )
            reward += reward_config["field_following"] * field_alignment
            
        return float(reward)
    
    def _compute_field_alignment(
        self,
        position: np.ndarray,
        action: np.ndarray,
        gr_field: torch.Tensor
    ) -> float:
        """Compute alignment between action and GR field gradient."""
        
        # Sample field at current position
        # This is a simplified version - actual implementation would
        # properly sample from the field tensor
        
        # Convert position to grid coordinates
        grid_x = int((position[0] + 10) / 20 * gr_field.shape[1])
        grid_y = int((position[2] + 10) / 20 * gr_field.shape[2])
        
        # Clamp to valid range
        grid_x = np.clip(grid_x, 0, gr_field.shape[1] - 1)
        grid_y = np.clip(grid_y, 0, gr_field.shape[2] - 1)
        
        # Compute field gradient
        if grid_x > 0 and grid_x < gr_field.shape[1] - 1:
            dx = gr_field[0, grid_x + 1, grid_y, 0] - gr_field[0, grid_x - 1, grid_y, 0]
        else:
            dx = 0.0
            
        if grid_y > 0 and grid_y < gr_field.shape[2] - 1:
            dy = gr_field[0, grid_x, grid_y + 1, 0] - gr_field[0, grid_x, grid_y - 1, 0]
        else:
            dy = 0.0
            
        # Normalize gradient
        gradient = np.array([dx.item(), dy.item()])
        gradient_norm = np.linalg.norm(gradient)
        
        if gradient_norm > 0:
            gradient = gradient / gradient_norm
            
        # Compute alignment with action direction
        action_dir = action[:2] / (np.linalg.norm(action[:2]) + 1e-8)
        alignment = np.dot(action_dir, gradient)
        
        return float(np.clip(alignment, -1.0, 1.0))
    
    def _add_advanced_observations(self, observations: Dict) -> Dict:
        """Add advanced observations for research."""
        
        # Add agent state
        agent_state = self._env.sim.get_agent_state()
        observations["agent_position"] = agent_state.position
        observations["agent_rotation"] = quaternion_to_list(agent_state.rotation)
        
        # Add goal information
        if hasattr(self._env.task, "goal"):
            observations["goal_position"] = self._env.task.goal.position
            
        # Add shortest path information
        if self.shortest_path_follower is not None:
            try:
                path = self.shortest_path_follower.get_path(
                    self._env.sim.get_agent_state().position,
                    self._env.task.goal.position
                )
                observations["shortest_path_length"] = len(path)
                observations["shortest_path_next_action"] = self.shortest_path_follower.get_next_action()
            except (RuntimeError, AttributeError, Exception) as e:
                observations["shortest_path_length"] = -1
                observations["shortest_path_next_action"] = 0
        else:
            observations["shortest_path_length"] = -1
            observations["shortest_path_next_action"] = 0
            
        # Add semantic information if available
        if "semantic" in observations:
            observations["semantic_categories"] = np.unique(observations["semantic"])
            
        return observations
    
    def _apply_sensor_noise(self, observations: Dict) -> Dict:
        """Apply realistic sensor noise for robustness testing."""
        
        # RGB noise
        if "rgb" in observations:
            # Gaussian noise
            noise = np.random.normal(0, 0.02, observations["rgb"].shape)
            observations["rgb"] = np.clip(observations["rgb"] + noise, 0, 1)
            
            # Random pixel dropout
            dropout_mask = np.random.random(observations["rgb"].shape[:2]) > 0.98
            observations["rgb"][dropout_mask] = 0
            
        # Depth noise
        if "depth" in observations:
            # Gaussian noise proportional to depth
            noise = np.random.normal(0, 0.01 * observations["depth"])
            observations["depth"] = np.maximum(observations["depth"] + noise, 0)
            
            # Depth holes (sensor failures)
            hole_mask = np.random.random(observations["depth"].shape) > 0.95
            observations["depth"][hole_mask] = 0
            
        return observations
    
    def _update_curriculum(self):
        """Update curriculum learning stage based on episode count."""
        
        if not self.adv_config.curriculum_enabled:
            return
            
        stages = self.adv_config.curriculum_stages
        
        for i, stage in enumerate(stages):
            if stage["episodes"] == -1 or self.total_episodes < stage["episodes"]:
                if self.curriculum_stage != i:
                    self.curriculum_stage = i
                    logger.info(f"Curriculum advanced to stage {i}: {stage}")
                    
                    # Update environment difficulty
                    self._apply_curriculum_stage(stage)
                break
                
    def _apply_curriculum_stage(self, stage: Dict):
        """Apply curriculum stage settings to environment."""
        
        # Update geodesic distance limit
        if "max_distance" in stage:
            self._env._config.defrost()
            self._env._config.task.geodesic_distance_limit = stage["max_distance"]
            self._env._config.freeze()
            
    def _get_info(self, observations: Dict) -> Dict:
        """Get comprehensive info dictionary."""
        
        info = {}
        
        # Add all metrics
        metrics = self._env.get_metrics()
        for key, value in metrics.items():
            info[key] = value
            
        # Add curriculum info
        info["curriculum_stage"] = self.curriculum_stage
        info["total_episodes"] = self.total_episodes
        
        # Add agent state
        agent_state = self._env.sim.get_agent_state()
        info["agent_position"] = agent_state.position
        info["agent_rotation"] = quaternion_to_list(agent_state.rotation)
        
        # Add scene info
        info["scene_id"] = self._env.current_episode.scene_id
        info["episode_id"] = self._env.current_episode.episode_id
        
        return info
    
    def set_gr_field(self, gr_field: torch.Tensor):
        """Set current GR field for reward computation."""
        self.current_gr_field = gr_field
        
    def render(self) -> Optional[np.ndarray]:
        """Render environment for visualization."""
        
        if self.render_mode is None:
            return None
            
        if self.render_mode == "rgb_array":
            return self._env.sim.get_sensor_observations()["rgb"]
        elif self.render_mode == "human":
            # Display using cv2 or matplotlib
            rgb = self._env.sim.get_sensor_observations()["rgb"]
            cv2.imshow("Habitat", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            return rgb
            
    def close(self):
        """Clean up environment."""
        if hasattr(self, "_env"):
            self._env.close()
        if self.video_writer is not None:
            self.video_writer.release()


class MetricsTracker:
    """Advanced metrics tracking for research analysis."""
    
    def __init__(self):
        self.episode_metrics = defaultdict(list)
        self.cumulative_metrics = defaultdict(float)
        
    def reset_episode(self):
        """Reset metrics for new episode."""
        self.episode_metrics.clear()
        
    def update(self, observations: Dict, reward: float, done: bool):
        """Update metrics with step information."""
        
        # Track rewards
        self.episode_metrics["rewards"].append(reward)
        
        # Track positions for trajectory analysis
        if "agent_position" in observations:
            self.episode_metrics["trajectory"].append(
                observations["agent_position"].copy()
            )
            
        # Track collisions
        if "collisions" in observations:
            self.episode_metrics["collisions"].append(
                observations["collisions"].get("is_collision", False)
            )
            
    def get_episode_summary(self) -> Dict:
        """Get summary statistics for episode."""
        
        summary = {
            "total_reward": sum(self.episode_metrics["rewards"]),
            "episode_length": len(self.episode_metrics["rewards"]),
            "num_collisions": sum(self.episode_metrics["collisions"]),
            "trajectory_length": self._compute_trajectory_length(),
            "trajectory_smoothness": self._compute_trajectory_smoothness()
        }
        
        return summary
    
    def _compute_trajectory_length(self) -> float:
        """Compute total trajectory length."""
        
        trajectory = self.episode_metrics.get("trajectory", [])
        if len(trajectory) < 2:
            return 0.0
            
        length = 0.0
        for i in range(1, len(trajectory)):
            length += np.linalg.norm(trajectory[i] - trajectory[i-1])
            
        return float(length)
    
    def _compute_trajectory_smoothness(self) -> float:
        """Compute trajectory smoothness (lower is smoother)."""
        
        trajectory = self.episode_metrics.get("trajectory", [])
        if len(trajectory) < 3:
            return 0.0
            
        smoothness = 0.0
        for i in range(2, len(trajectory)):
            v1 = trajectory[i-1] - trajectory[i-2]
            v2 = trajectory[i] - trajectory[i-1]
            smoothness += np.linalg.norm(v2 - v1)
            
        return float(smoothness / (len(trajectory) - 2))

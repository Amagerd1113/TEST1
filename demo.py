#!/usr/bin/env python3
"""
Interactive demo for VLA-GR navigation framework.
Provides real-time visualization and control interface.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional, List

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import habitat
try:
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
except ImportError:
    # Fallback for different habitat versions
    try:
        from habitat_sim import SimulatorActions as HabitatSimActions
    except ImportError:
        # Define basic actions if neither import works
        class HabitatSimActions:
            STOP = 0
            MOVE_FORWARD = 1
            TURN_LEFT = 2
            TURN_RIGHT = 3

# Setup path
import sys
sys.path.append(str(Path(__file__).parent))

# Setup logger first
logger = logging.getLogger(__name__)

# Import VLA-GR agent (use v2 version if available)
try:
    from src.core.vla_gr_agent import ConferenceVLAGRAgent as VLAGRAgent, VLAGRStateV2 as VLAGRState
except ImportError:
    try:
        from src.core.vla_gr_agent import VLAGRAgent, VLAGRState
    except ImportError:
        logger.error("Could not import VLA-GR agent")
        VLAGRAgent = None
        VLAGRState = None

try:
    from src.utils.visualization import Visualizer
except ImportError:
    Visualizer = None

try:
    from src.datasets.habitat_dataset import HabitatNavigationDataset
except ImportError:
    logger.warning("Could not import HabitatNavigationDataset")
    HabitatNavigationDataset = None


class VLAGRDemo:
    """Interactive demonstration of VLA-GR navigation."""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.yaml",
        device: str = "cuda",
        visualize: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.visualize = visualize
        
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.agent = self._load_model(model_path)
        self.agent.eval()
        
        # Setup Habitat simulator
        self.simulator = self._setup_simulator()
        
        # Setup visualizer
        if self.visualize:
            self.visualizer = Visualizer(self.config)
            self.setup_visualization()
            
        # Navigation state
        self.current_position = None
        self.current_orientation = None
        self.current_velocity = torch.zeros(3, device=self.device)
        self.trajectory_history = []
        
        # Performance tracking
        self.metrics = {
            'steps': 0,
            'distance_traveled': 0.0,
            'collisions': 0,
            'success': False
        }
        
    def _load_model(self, model_path: str) -> VLAGRAgent:
        """Load pre-trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = VLAGRAgent(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    
    def _setup_simulator(self):
        """Initialize Habitat simulator."""
        from habitat.config.default import get_config
        from habitat.sims import make_sim

        try:
            habitat_config = get_config(config_paths="configs/tasks/pointnav_gibson.yaml")
        except:
            # Fallback: create basic config
            habitat_config = get_config()

        habitat_config.defrost()
        habitat_config.SIMULATOR.RGB_SENSOR.WIDTH = 640
        habitat_config.SIMULATOR.RGB_SENSOR.HEIGHT = 480
        habitat_config.SIMULATOR.DEPTH_SENSOR.WIDTH = 640
        habitat_config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 480
        habitat_config.freeze()

        return make_sim(
            id_sim=habitat_config.SIMULATOR.TYPE,
            config=habitat_config.SIMULATOR
        )
    
    def setup_visualization(self):
        """Setup visualization windows."""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Configure subplots
        self.axes[0, 0].set_title("RGB Input")
        self.axes[0, 1].set_title("Depth Map")
        self.axes[0, 2].set_title("Affordance Map")
        self.axes[1, 0].set_title("GR Field")
        self.axes[1, 1].set_title("Planned Path")
        self.axes[1, 2].set_title("Top-Down View")
        
        for ax in self.axes.flat:
            ax.axis('off')
            
        plt.tight_layout()
        
    def reset_episode(
        self,
        scene: Optional[str] = None,
        start_position: Optional[List[float]] = None,
        goal_position: Optional[List[float]] = None
    ):
        """Reset for new navigation episode."""
        
        # Reset simulator
        self.simulator.reset()
        
        if start_position is None:
            start_position = self.simulator.sample_navigable_point()
            
        self.simulator.set_agent_state(
            position=start_position,
            rotation=[0, 0, 0, 1]
        )
        
        # Reset state
        self.current_position = torch.tensor(start_position, device=self.device)
        self.current_orientation = torch.tensor([0, 0, 0, 1], device=self.device)
        self.current_velocity = torch.zeros(3, device=self.device)
        self.trajectory_history = [start_position]
        
        # Reset metrics
        self.metrics = {
            'steps': 0,
            'distance_traveled': 0.0,
            'collisions': 0,
            'success': False
        }
        
        # Reset agent
        self.agent.reset()
        
        logger.info(f"Episode reset at position {start_position}")
        
    def step(self, instruction: str) -> Dict:
        """Execute one navigation step."""
        
        # Get observations
        obs = self.simulator.get_sensor_observations()
        
        # Preprocess observations
        rgb = torch.from_numpy(obs['rgb']).permute(2, 0, 1).float() / 255.0
        rgb = rgb.unsqueeze(0).to(self.device)
        
        depth = torch.from_numpy(obs['depth']).unsqueeze(0).unsqueeze(0).float()
        depth = depth.to(self.device)
        
        # Create state
        state = VLAGRState(
            rgb_image=rgb,
            depth_map=depth,
            language_instruction=[instruction],
            position=self.current_position.unsqueeze(0),
            orientation=self.current_orientation.unsqueeze(0),
            velocity=self.current_velocity.unsqueeze(0)
        )
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.agent(state, deterministic=True)
            
        # Extract action
        action = outputs['actions'][0].cpu().numpy()
        
        # Execute action in simulator
        self._execute_action(action)
        
        # Update metrics
        self.metrics['steps'] += 1
        
        # Visualization
        if self.visualize:
            self._update_visualization(obs, outputs)
            
        return {
            'action': action,
            'outputs': outputs,
            'metrics': self.metrics
        }
    
    def _execute_action(self, action: np.ndarray):
        """Execute action in simulator."""
        
        # Convert continuous action to discrete simulator actions
        velocity = action[:3]
        rotation = action[3:7]
        
        # Threshold for movement
        forward_threshold = 0.5
        turn_threshold = 0.3
        
        if velocity[0] > forward_threshold:
            # Move forward
            self.simulator.step(HabitatSimActions.MOVE_FORWARD)
        elif velocity[0] < -forward_threshold:
            # Move backward (if supported)
            pass
            
        if rotation[1] > turn_threshold:
            # Turn right
            self.simulator.step(HabitatSimActions.TURN_RIGHT)
        elif rotation[1] < -turn_threshold:
            # Turn left
            self.simulator.step(HabitatSimActions.TURN_LEFT)
            
        # Update position
        agent_state = self.simulator.get_agent_state()
        new_position = agent_state.position
        
        # Check for collision
        distance_moved = np.linalg.norm(
            np.array(new_position) - self.current_position.cpu().numpy()
        )
        
        if distance_moved < 0.01:  # Threshold for collision detection
            self.metrics['collisions'] += 1
            logger.warning("Collision detected!")
            
        # Update state
        self.current_position = torch.tensor(new_position, device=self.device)
        self.current_orientation = torch.tensor(
            agent_state.rotation,
            device=self.device
        )
        
        # Update trajectory
        self.trajectory_history.append(new_position.tolist())
        self.metrics['distance_traveled'] += distance_moved
        
    def _update_visualization(self, obs: Dict, outputs: Dict):
        """Update visualization plots."""
        
        # RGB image
        self.axes[0, 0].clear()
        self.axes[0, 0].imshow(obs['rgb'])
        self.axes[0, 0].set_title("RGB Input")
        self.axes[0, 0].axis('off')
        
        # Depth map
        self.axes[0, 1].clear()
        self.axes[0, 1].imshow(obs['depth'], cmap='viridis')
        self.axes[0, 1].set_title("Depth Map")
        self.axes[0, 1].axis('off')
        
        # Affordance map
        if 'affordance_map' in outputs:
            affordance = outputs['affordance_map'][0].cpu().numpy()
            affordance_vis = affordance.sum(axis=-1)
            self.axes[0, 2].clear()
            self.axes[0, 2].imshow(affordance_vis, cmap='hot')
            self.axes[0, 2].set_title("Affordance Map")
            self.axes[0, 2].axis('off')
            
        # GR field
        if 'gr_field' in outputs:
            gr_field = outputs['gr_field'][0].cpu().numpy()
            field_vis = np.linalg.norm(gr_field, axis=-1)
            self.axes[1, 0].clear()
            self.axes[1, 0].imshow(field_vis, cmap='plasma')
            self.axes[1, 0].set_title("GR Field Magnitude")
            self.axes[1, 0].axis('off')
            
        # Planned path
        if 'planned_path' in outputs:
            path = outputs['planned_path'][0].cpu().numpy()
            self.axes[1, 1].clear()
            self.axes[1, 1].plot(path[:, 0], path[:, 2], 'b-', linewidth=2)
            self.axes[1, 1].scatter(path[0, 0], path[0, 2], c='g', s=100, label='Start')
            self.axes[1, 1].scatter(path[-1, 0], path[-1, 2], c='r', s=100, label='Goal')
            self.axes[1, 1].set_title("Planned Path")
            self.axes[1, 1].legend()
            self.axes[1, 1].set_xlabel("X (m)")
            self.axes[1, 1].set_ylabel("Z (m)")
            self.axes[1, 1].grid(True)
            
        # Top-down view with trajectory
        if len(self.trajectory_history) > 1:
            traj = np.array(self.trajectory_history)
            self.axes[1, 2].clear()
            self.axes[1, 2].plot(traj[:, 0], traj[:, 2], 'g-', linewidth=2, alpha=0.7)
            self.axes[1, 2].scatter(traj[-1, 0], traj[-1, 2], c='b', s=100, label='Current')
            self.axes[1, 2].set_title(f"Trajectory (Steps: {self.metrics['steps']})")
            self.axes[1, 2].legend()
            self.axes[1, 2].set_xlabel("X (m)")
            self.axes[1, 2].set_ylabel("Z (m)")
            self.axes[1, 2].grid(True)
            
        plt.draw()
        plt.pause(0.001)
        
    def run_interactive(self):
        """Run interactive demonstration."""
        
        print("\n" + "="*50)
        print("VLA-GR Navigation Demo")
        print("="*50)
        print("\nCommands:")
        print("  - Type navigation instruction (e.g., 'Navigate to the red chair')")
        print("  - 'reset': Start new episode")
        print("  - 'auto': Run automatic navigation")
        print("  - 'quit': Exit demo")
        print("="*50 + "\n")
        
        # Reset for first episode
        self.reset_episode()
        
        while True:
            try:
                command = input("\nEnter command: ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'reset':
                    self.reset_episode()
                    print("Episode reset!")
                elif command.lower() == 'auto':
                    self.run_automatic()
                else:
                    # Use as navigation instruction
                    result = self.step(command)
                    print(f"Step {self.metrics['steps']}: Action taken")
                    print(f"  Distance traveled: {self.metrics['distance_traveled']:.2f}m")
                    print(f"  Collisions: {self.metrics['collisions']}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error during execution: {e}")
                
        print("\nDemo finished!")
        
    def run_automatic(self, instruction: str = "Navigate to the nearest chair"):
        """Run automatic navigation to completion."""
        
        print(f"\nRunning automatic navigation: '{instruction}'")
        
        max_steps = 100
        success_threshold = 0.2  # meters
        
        for step in range(max_steps):
            result = self.step(instruction)
            
            # Check if reached goal
            if 'goal_position' in result['outputs']:
                goal = result['outputs']['goal_position'][0].cpu().numpy()
                current = self.current_position.cpu().numpy()
                distance = np.linalg.norm(goal - current)
                
                if distance < success_threshold:
                    self.metrics['success'] = True
                    print(f"\n✓ Goal reached in {step} steps!")
                    break
                    
            # Display progress
            if step % 10 == 0:
                print(f"  Step {step}: Distance to goal: {distance:.2f}m")
                
        if not self.metrics['success']:
            print(f"\n✗ Failed to reach goal within {max_steps} steps")
            
        # Final statistics
        print(f"\nFinal Statistics:")
        print(f"  Total steps: {self.metrics['steps']}")
        print(f"  Distance traveled: {self.metrics['distance_traveled']:.2f}m")
        print(f"  Collisions: {self.metrics['collisions']}")
        print(f"  Success: {self.metrics['success']}")
        
    def benchmark(self, num_episodes: int = 100):
        """Run benchmark evaluation."""
        
        print(f"\nRunning benchmark with {num_episodes} episodes...")
        
        results = []
        
        for episode in range(num_episodes):
            self.reset_episode()
            
            # Random instruction
            instructions = [
                "Navigate to the chair",
                "Go to the table",
                "Find the door",
                "Move to the couch",
                "Reach the bed"
            ]
            instruction = np.random.choice(instructions)
            
            # Run episode
            self.run_automatic(instruction)
            
            # Store results
            results.append({
                'episode': episode,
                'success': self.metrics['success'],
                'steps': self.metrics['steps'],
                'distance': self.metrics['distance_traveled'],
                'collisions': self.metrics['collisions']
            })
            
            # Progress
            if (episode + 1) % 10 == 0:
                success_rate = np.mean([r['success'] for r in results])
                print(f"  Episodes {episode + 1}/{num_episodes}: Success rate = {success_rate:.2%}")
                
        # Compute final metrics
        success_rate = np.mean([r['success'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results if r['success']])
        avg_collisions = np.mean([r['collisions'] for r in results])
        
        print(f"\n{'='*50}")
        print(f"Benchmark Results:")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Avg Steps (successful): {avg_steps:.1f}")
        print(f"  Avg Collisions: {avg_collisions:.2f}")
        print(f"{'='*50}\n")
        
        return results


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="VLA-GR Navigation Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "auto", "benchmark"],
        default="interactive",
        help="Demo mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes for benchmark"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s"
    )
    
    # Create demo
    demo = VLAGRDemo(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        visualize=not args.no_viz
    )
    
    # Run demo
    if args.mode == "interactive":
        demo.run_interactive()
    elif args.mode == "auto":
        demo.reset_episode()
        demo.run_automatic()
    elif args.mode == "benchmark":
        demo.benchmark(num_episodes=args.num_episodes)


if __name__ == "__main__":
    main()

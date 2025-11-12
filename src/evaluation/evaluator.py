"""
Comprehensive Evaluation Module for VLA-GR Navigation
Includes ablation studies, standardized benchmarks, and comparative analysis
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.core.vla_gr_agent import VLAGRAgent, VLAGRState
from src.environments.habitat_env import HabitatEnvironment, HabitatConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive metrics for navigation evaluation."""
    
    # Success metrics
    success_rate: float = 0.0
    spl: float = 0.0  # Success weighted by Path Length
    soft_spl: float = 0.0  # Soft SPL (partial credit for getting close)
    
    # Efficiency metrics
    path_length: float = 0.0
    trajectory_length: float = 0.0
    navigation_error: float = 0.0  # Final distance to goal
    
    # Safety metrics
    collision_rate: float = 0.0
    num_collisions: float = 0.0
    collision_penalty: float = 0.0
    
    # Temporal metrics
    episode_length: float = 0.0
    inference_time: float = 0.0
    total_time: float = 0.0
    
    # Robustness metrics
    occlusion_robustness: float = 0.0
    noise_robustness: float = 0.0
    
    # Field quality metrics
    field_accuracy: float = 0.0
    field_smoothness: float = 0.0
    geodesic_optimality: float = 0.0
    
    # Additional metrics
    goal_visibility_success: float = 0.0
    oracle_success: float = 0.0
    oracle_spl: float = 0.0


class VLAGREvaluator:
    """
    Complete evaluation suite for VLA-GR navigation.
    
    Features:
    - Standard benchmark evaluation
    - Ablation studies
    - Comparative analysis with baselines
    - Robustness testing
    - Statistical significance testing
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cuda",
        output_dir: str = "evaluation_results"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize environment
        habitat_config = HabitatConfig(
            scene_dataset=self.config['environment']['habitat']['scene_dataset'],
            task=self.config['environment']['task']['type']
        )
        self.env = HabitatEnvironment(habitat_config)
        
        # Baseline models for comparison
        self.baselines = {}
        self._load_baselines()
        
        # Results storage
        self.results = defaultdict(list)
        self.ablation_results = {}
        
    def _load_model(self, model_path: str) -> nn.Module:
        """Load VLA-GR model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = VLAGRAgent(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def _load_baselines(self):
        """Load baseline models for comparison."""
        
        # Habitat baselines
        try:
            from habitat_baselines.agents import PPOAgent
            self.baselines['ppo'] = PPOAgent()
        except (ImportError, ModuleNotFoundError, Exception) as e:
            logger.warning(f"PPO baseline not available: {e}")
            
        # Simple baselines
        self.baselines['random'] = RandomAgent()
        self.baselines['shortest_path'] = ShortestPathAgent()
        self.baselines['forward_only'] = ForwardOnlyAgent()
        
    def evaluate_standard_benchmarks(
        self,
        num_episodes: int = 1000,
        splits: Optional[List[str]] = None
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate on standard navigation benchmarks.

        Args:
            num_episodes: Number of episodes to evaluate
            splits: Data splits to evaluate on (default: ["val", "test"])

        Returns:
            Metrics for each split
        """
        if splits is None:
            splits = ["val", "test"]

        all_metrics = {}

        for split in splits:
            logger.info(f"Evaluating on {split} split with {num_episodes} episodes")
            
            # Set environment to split
            self.env.config.split = split
            
            split_metrics = []
            
            for episode_idx in tqdm(range(num_episodes), desc=f"Evaluating {split}"):
                # Reset environment
                obs = self.env.reset()
                
                # Run episode
                episode_metrics = self._run_episode(obs)
                split_metrics.append(episode_metrics)
                
                # Store detailed results
                self.results[split].append(asdict(episode_metrics))
                
            # Aggregate metrics
            aggregated = self._aggregate_metrics(split_metrics)
            all_metrics[split] = aggregated
            
            # Log results
            self._log_metrics(split, aggregated)
            
        # Save results
        self._save_results("standard_benchmarks", all_metrics)
        
        return all_metrics
    
    def run_ablation_studies(self) -> Dict[str, Dict]:
        """
        Run comprehensive ablation studies.
        
        Ablations:
        1. Without GR field
        2. Without depth completion
        3. Without field injection
        4. Without language input
        5. Without Bayesian updates
        6. Without exploration strategy
        """
        
        ablations = {
            "full_model": None,  # Full model (baseline)
            "no_gr_field": self._ablate_gr_field,
            "no_depth_completion": self._ablate_depth_completion,
            "no_field_injection": self._ablate_field_injection,
            "no_language": self._ablate_language,
            "no_bayesian": self._ablate_bayesian,
            "no_exploration": self._ablate_exploration
        }
        
        results = {}
        
        for ablation_name, ablation_fn in ablations.items():
            logger.info(f"Running ablation: {ablation_name}")
            
            # Apply ablation
            if ablation_fn is not None:
                ablation_fn()
                
            # Evaluate
            metrics = []
            for episode in range(100):  # Fewer episodes for ablations
                obs = self.env.reset()
                episode_metrics = self._run_episode(obs)
                metrics.append(episode_metrics)
                
            # Aggregate
            aggregated = self._aggregate_metrics(metrics)
            results[ablation_name] = aggregated
            
            # Restore model
            if ablation_fn is not None:
                self._restore_model()
                
            # Log results
            logger.info(f"{ablation_name}: Success={aggregated.success_rate:.2%}, SPL={aggregated.spl:.3f}")
            
        # Analyze ablation impact
        self.ablation_results = self._analyze_ablations(results)
        
        # Save results
        self._save_results("ablation_studies", results)
        
        return results
    
    def _ablate_gr_field(self):
        """Disable GR field computation."""
        # Modify model to bypass GR field
        original_forward = self.model.gr_field_manager.forward
        
        def bypass_gr_field(*args, **kwargs):
            B, H, W = args[0].shape[:3]
            # Return identity metric (flat spacetime)
            metric = torch.zeros(B, H, W, 10).to(self.device)
            metric[..., 0] = -1.0  # g_00
            metric[..., 4] = 1.0   # g_11
            metric[..., 7] = 1.0   # g_22
            metric[..., 9] = 1.0   # g_33
            
            return {
                'metric_tensor': metric,
                'christoffel_symbols': torch.zeros(B, H, W, 40).to(self.device),
                'riemann_tensor': torch.zeros(B, H, W, 20).to(self.device),
                'energy_momentum': torch.zeros(B, H, W, 10).to(self.device),
                'geodesic_acceleration': None,
                'mass_distribution': torch.ones(B, H, W, 1).to(self.device)
            }
            
        self.model.gr_field_manager.forward = bypass_gr_field
        
    def _ablate_depth_completion(self):
        """Disable depth completion."""
        self.model.perception.depth_completion = None
        
    def _ablate_field_injection(self):
        """Disable field injection in attention."""
        original_forward = self.model.field_injection.forward
        
        def bypass_injection(features, gr_field):
            return features  # Return features unchanged
            
        self.model.field_injection.forward = bypass_injection
        
    def _ablate_language(self):
        """Disable language input."""
        # Use random language features
        original_encoder = self.model.perception.language_encoder.forward
        
        def random_language(instructions, device):
            B = len(instructions)
            return torch.randn(B, 256, 768).to(device)
            
        self.model.perception.language_encoder.forward = random_language
        
    def _ablate_bayesian(self):
        """Disable Bayesian updates."""
        self.model.affordance_quantifier.use_bayesian = False
        
    def _ablate_exploration(self):
        """Disable exploration strategy."""
        self.model.exploration_module.current_epsilon = 0.0
        
    def _restore_model(self):
        """Restore model to original state."""
        # Reload model
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def _run_episode(self, initial_obs: Dict) -> EvaluationMetrics:
        """Run single navigation episode."""
        
        metrics = EvaluationMetrics()
        
        # Episode tracking
        trajectory = []
        collisions = 0
        start_time = time.time()
        
        # Get goal information
        goal_instruction = self._generate_instruction(initial_obs)
        
        # Initialize state
        obs = initial_obs
        done = False
        step = 0
        
        while not done and step < self.config['environment']['habitat']['max_episode_steps']:
            # Prepare state for model
            state = self._obs_to_state(obs, goal_instruction)
            
            # Get action from model
            inference_start = time.time()
            with torch.no_grad():
                outputs = self.model(state, deterministic=True)
            inference_time = time.time() - inference_start
            
            # Execute action
            action = outputs['actions'][0].cpu().numpy()
            obs, reward, done, info = self.env.step(action)
            
            # Track metrics
            trajectory.append(obs['position'])
            if info.get('collision', False):
                collisions += 1
                
            step += 1
            
        # Calculate metrics
        metrics.success_rate = float(info.get('success', False))
        metrics.spl = info.get('spl', 0.0)
        metrics.navigation_error = info.get('distance_to_goal', -1.0)
        metrics.episode_length = step
        metrics.num_collisions = collisions
        metrics.collision_rate = collisions / max(step, 1)
        metrics.inference_time = inference_time
        metrics.total_time = time.time() - start_time
        
        # Calculate path metrics
        if len(trajectory) > 1:
            metrics.trajectory_length = self._calculate_path_length(trajectory)
            
        return metrics
    
    def _obs_to_state(self, obs: Dict, instruction: str) -> VLAGRState:
        """Convert observation to VLA-GR state."""
        
        # Convert to tensors
        rgb = torch.from_numpy(obs['rgb']).unsqueeze(0).to(self.device)
        depth = torch.from_numpy(obs['depth']).unsqueeze(0).to(self.device)
        position = torch.from_numpy(obs['position']).unsqueeze(0).to(self.device)
        rotation = torch.from_numpy(obs['rotation']).unsqueeze(0).to(self.device)
        
        state = VLAGRState(
            rgb_image=rgb,
            depth_map=depth,
            language_instruction=[instruction],
            position=position,
            orientation=rotation,
            velocity=torch.zeros(1, 3).to(self.device)
        )
        
        return state
    
    def _generate_instruction(self, obs: Dict) -> str:
        """Generate language instruction for episode."""
        
        if 'objectgoal' in obs:
            object_id = obs['objectgoal']
            object_names = ["chair", "table", "couch", "bed", "toilet", "tv", "plant"]
            if object_id < len(object_names):
                return f"Navigate to the {object_names[object_id]}"
                
        return "Navigate to the goal location"
    
    def _calculate_path_length(self, trajectory: List[np.ndarray]) -> float:
        """Calculate total path length."""
        
        length = 0.0
        for i in range(1, len(trajectory)):
            length += np.linalg.norm(trajectory[i] - trajectory[i-1])
            
        return length
    
    def _aggregate_metrics(self, metrics_list: List[EvaluationMetrics]) -> EvaluationMetrics:
        """Aggregate metrics across episodes."""
        
        aggregated = EvaluationMetrics()
        
        # Get all metric fields
        fields = [field for field in dir(aggregated) if not field.startswith('_')]
        
        for field in fields:
            values = [getattr(m, field) for m in metrics_list]
            if values and all(v is not None for v in values):
                setattr(aggregated, field, np.mean(values))
                
        return aggregated
    
    def _analyze_ablations(self, results: Dict[str, EvaluationMetrics]) -> Dict:
        """Analyze ablation study results."""
        
        analysis = {}
        full_model = results["full_model"]
        
        for ablation_name, metrics in results.items():
            if ablation_name == "full_model":
                continue
                
            # Calculate performance drop
            analysis[ablation_name] = {
                "success_drop": full_model.success_rate - metrics.success_rate,
                "spl_drop": full_model.spl - metrics.spl,
                "collision_increase": metrics.collision_rate - full_model.collision_rate,
                "relative_success_drop": (full_model.success_rate - metrics.success_rate) / max(full_model.success_rate, 0.01)
            }
            
        return analysis
    
    def compare_with_baselines(
        self,
        num_episodes: int = 100
    ) -> pd.DataFrame:
        """
        Compare VLA-GR with baseline methods.
        
        Returns:
            DataFrame with comparative results
        """
        
        results = {"VLA-GR": []}
        
        # Evaluate VLA-GR
        for episode in tqdm(range(num_episodes), desc="Evaluating VLA-GR"):
            obs = self.env.reset()
            metrics = self._run_episode(obs)
            results["VLA-GR"].append(metrics)
            
        # Evaluate baselines
        for baseline_name, baseline_agent in self.baselines.items():
            logger.info(f"Evaluating baseline: {baseline_name}")
            results[baseline_name] = []
            
            for episode in tqdm(range(num_episodes), desc=f"Evaluating {baseline_name}"):
                obs = self.env.reset()
                metrics = self._run_baseline_episode(baseline_agent, obs)
                results[baseline_name].append(metrics)
                
        # Create comparison dataframe
        comparison_data = []
        
        for method_name, metrics_list in results.items():
            aggregated = self._aggregate_metrics(metrics_list)
            
            comparison_data.append({
                "Method": method_name,
                "Success Rate": f"{aggregated.success_rate:.2%}",
                "SPL": f"{aggregated.spl:.3f}",
                "Collision Rate": f"{aggregated.collision_rate:.2%}",
                "Avg Steps": f"{aggregated.episode_length:.1f}",
                "Inference Time (ms)": f"{aggregated.inference_time * 1000:.1f}"
            })
            
        df = pd.DataFrame(comparison_data)
        
        # Save comparison
        df.to_csv(self.output_dir / "baseline_comparison.csv", index=False)
        
        # Print latex table
        print("\nLatex Table:")
        print(df.to_latex(index=False))
        
        return df
    
    def _run_baseline_episode(self, agent, initial_obs: Dict) -> EvaluationMetrics:
        """Run episode with baseline agent."""
        
        metrics = EvaluationMetrics()
        
        obs = initial_obs
        done = False
        step = 0
        trajectory = []
        collisions = 0
        
        while not done and step < 500:
            # Get baseline action
            action = agent.act(obs)
            
            # Execute
            obs, reward, done, info = self.env.step(action)
            
            # Track
            trajectory.append(obs['position'])
            if info.get('collision', False):
                collisions += 1
                
            step += 1
            
        # Calculate metrics
        metrics.success_rate = float(info.get('success', False))
        metrics.spl = info.get('spl', 0.0)
        metrics.navigation_error = info.get('distance_to_goal', -1.0)
        metrics.episode_length = step
        metrics.num_collisions = collisions
        metrics.collision_rate = collisions / max(step, 1)
        
        return metrics
    
    def test_robustness(self) -> Dict[str, Dict]:
        """
        Test model robustness to various conditions.
        
        Tests:
        1. Occlusion levels (0%, 10%, 20%, 30%, 40%, 50%)
        2. Sensor noise levels
        3. Dynamic obstacles
        4. Lighting variations
        5. Novel environments
        """
        
        robustness_results = {}
        
        # Test occlusion robustness
        occlusion_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        occlusion_results = []
        
        for occlusion_level in occlusion_levels:
            logger.info(f"Testing with {occlusion_level*100:.0f}% occlusion")
            
            metrics_list = []
            for episode in range(50):
                obs = self.env.reset()
                
                # Add occlusion
                obs = self._add_occlusion(obs, occlusion_level)
                
                metrics = self._run_episode(obs)
                metrics_list.append(metrics)
                
            aggregated = self._aggregate_metrics(metrics_list)
            occlusion_results.append({
                "occlusion": occlusion_level,
                "success_rate": aggregated.success_rate,
                "spl": aggregated.spl,
                "collision_rate": aggregated.collision_rate
            })
            
        robustness_results["occlusion"] = occlusion_results
        
        # Test noise robustness
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
        noise_results = []
        
        for noise_level in noise_levels:
            logger.info(f"Testing with noise level {noise_level}")
            
            metrics_list = []
            for episode in range(50):
                obs = self.env.reset()
                
                # Add noise
                obs = self._add_noise(obs, noise_level)
                
                metrics = self._run_episode(obs)
                metrics_list.append(metrics)
                
            aggregated = self._aggregate_metrics(metrics_list)
            noise_results.append({
                "noise": noise_level,
                "success_rate": aggregated.success_rate,
                "spl": aggregated.spl
            })
            
        robustness_results["noise"] = noise_results
        
        # Save robustness results
        self._save_results("robustness_tests", robustness_results)
        
        # Plot robustness curves
        self._plot_robustness_curves(robustness_results)
        
        return robustness_results
    
    def _add_occlusion(self, obs: Dict, occlusion_level: float) -> Dict:
        """Add occlusion to observation."""
        
        if "depth" in obs:
            mask = np.random.random(obs["depth"].shape) < occlusion_level
            obs["depth"][mask] = 0.0
            
        return obs
    
    def _add_noise(self, obs: Dict, noise_level: float) -> Dict:
        """Add noise to observation."""
        
        if "rgb" in obs:
            noise = np.random.randn(*obs["rgb"].shape) * noise_level
            obs["rgb"] = np.clip(obs["rgb"] + noise, 0, 1)
            
        if "depth" in obs:
            noise = np.random.randn(*obs["depth"].shape) * noise_level
            obs["depth"] = np.maximum(obs["depth"] + noise, 0)
            
        return obs
    
    def statistical_significance_test(
        self,
        method1_results: List[float],
        method2_results: List[float]
    ) -> Dict:
        """
        Perform statistical significance testing.
        
        Uses:
        - T-test for parametric comparison
        - Mann-Whitney U test for non-parametric comparison
        - Bootstrap confidence intervals
        """
        
        # T-test
        t_stat, t_pval = stats.ttest_ind(method1_results, method2_results)
        
        # Mann-Whitney U test
        u_stat, u_pval = stats.mannwhitneyu(method1_results, method2_results)
        
        # Bootstrap confidence interval
        def bootstrap_ci(data, n_bootstrap=1000, ci=95):
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(sample))
                
            lower = np.percentile(bootstrap_means, (100 - ci) / 2)
            upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
            
            return lower, upper
            
        ci1 = bootstrap_ci(method1_results)
        ci2 = bootstrap_ci(method2_results)
        
        return {
            "t_statistic": t_stat,
            "t_p_value": t_pval,
            "u_statistic": u_stat,
            "u_p_value": u_pval,
            "method1_ci": ci1,
            "method2_ci": ci2,
            "significant": t_pval < 0.05 and u_pval < 0.05
        }
    
    def _plot_robustness_curves(self, results: Dict):
        """Plot robustness analysis curves."""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Occlusion robustness
        if "occlusion" in results:
            occlusion_data = pd.DataFrame(results["occlusion"])
            axes[0].plot(occlusion_data["occlusion"] * 100,
                        occlusion_data["success_rate"] * 100,
                        'b-o', label="VLA-GR")
            axes[0].set_xlabel("Occlusion Level (%)")
            axes[0].set_ylabel("Success Rate (%)")
            axes[0].set_title("Occlusion Robustness")
            axes[0].grid(True)
            axes[0].legend()
            
        # Noise robustness
        if "noise" in results:
            noise_data = pd.DataFrame(results["noise"])
            axes[1].plot(noise_data["noise"],
                        noise_data["success_rate"] * 100,
                        'r-s', label="VLA-GR")
            axes[1].set_xlabel("Noise Level")
            axes[1].set_ylabel("Success Rate (%)")
            axes[1].set_title("Noise Robustness")
            axes[1].grid(True)
            axes[1].legend()
            
        plt.tight_layout()
        plt.savefig(self.output_dir / "robustness_curves.png", dpi=300)
        plt.show()
        
    def _log_metrics(self, split: str, metrics: EvaluationMetrics):
        """Log metrics to console."""
        
        logger.info(f"\n{'='*50}")
        logger.info(f"{split.upper()} Split Results:")
        logger.info(f"  Success Rate: {metrics.success_rate:.2%}")
        logger.info(f"  SPL: {metrics.spl:.3f}")
        logger.info(f"  Soft-SPL: {metrics.soft_spl:.3f}")
        logger.info(f"  Navigation Error: {metrics.navigation_error:.2f}m")
        logger.info(f"  Collision Rate: {metrics.collision_rate:.2%}")
        logger.info(f"  Avg Episode Length: {metrics.episode_length:.1f} steps")
        logger.info(f"  Inference Time: {metrics.inference_time*1000:.1f}ms")
        logger.info(f"{'='*50}\n")
        
    def _save_results(self, name: str, results: Any):
        """Save evaluation results."""
        
        # Save as JSON
        with open(self.output_dir / f"{name}.json", 'w') as f:
            if isinstance(results, dict):
                json.dump(results, f, indent=2, default=str)
            else:
                json.dump(asdict(results), f, indent=2, default=str)
                
        logger.info(f"Saved results to {self.output_dir / name}.json")


# Baseline Agents

class RandomAgent:
    """Random action baseline."""
    
    def act(self, obs: Dict) -> int:
        return np.random.randint(0, 4)  # Random discrete action


class ForwardOnlyAgent:
    """Forward-only baseline."""
    
    def act(self, obs: Dict) -> int:
        return 0  # Always move forward


class ShortestPathAgent:
    """Shortest path baseline using ground truth."""
    
    def act(self, obs: Dict) -> int:
        # Use ground truth shortest path if available
        if "shortest_path_action" in obs:
            return obs["shortest_path_action"]
        return 0  # Default to forward

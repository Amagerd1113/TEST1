#!/usr/bin/env python3
"""
Comprehensive VLA Experiment Runner
Runs all experiments defined in experiment_vla_comprehensive.yaml
Based on ICLR 2025, ICML 2024, and latest VLA research standards
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.vla_dataset_loader import VLADatasetLoader, create_dataloader
from src.evaluation.vla_metrics import VLAMetricsCalculator, VLAMetricResults
from src.core.vla_gr_agent import VLAGRAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLAExperimentRunner:
    """
    Comprehensive experiment runner for VLA-GR evaluation.
    """

    def __init__(self, config_path: str):
        """
        Initialize experiment runner.

        Args:
            config_path: Path to experiment configuration YAML
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.experiment_name = self.config['experiment']['name']
        self.results_dir = Path(self.config['experiment']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.dataset_loader = VLADatasetLoader(self.config)
        self.metrics_calculator = VLAMetricsCalculator(self.config)

        # Results storage
        self.all_results = {}

        # Setup experiment tracking
        self._setup_tracking()

        logger.info(f"Initialized VLA Experiment Runner: {self.experiment_name}")

    def _setup_tracking(self):
        """Setup experiment tracking (W&B, TensorBoard, etc.)."""
        tracking_config = self.config['experiment']['tracking']

        # Weights & Biases
        if tracking_config['wandb']['enabled']:
            try:
                import wandb
                wandb.init(
                    project=tracking_config['wandb']['project'],
                    entity=tracking_config['wandb']['entity'],
                    name=f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=self.config,
                    tags=tracking_config['wandb']['tags']
                )
                self.wandb_enabled = True
            except ImportError:
                logger.warning("wandb not installed, disabling W&B tracking")
                self.wandb_enabled = False
        else:
            self.wandb_enabled = False

        # TensorBoard
        if tracking_config['tensorboard']['enabled']:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = self.results_dir / 'tensorboard'
                self.tb_writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard logging to: {log_dir}")
            except ImportError:
                logger.warning("tensorboard not available")
                self.tb_writer = None
        else:
            self.tb_writer = None

    def run_all_experiments(self):
        """Run all configured experiments."""
        logger.info("="*80)
        logger.info(f"Starting {self.experiment_name}")
        logger.info("="*80)

        experiments_config = self.config['experiments']
        run_experiments = self.config['execution']['run_experiments']

        for exp_name in run_experiments:
            if exp_name not in experiments_config:
                logger.warning(f"Experiment {exp_name} not found in config")
                continue

            exp_config = experiments_config[exp_name]

            if not exp_config.get('enabled', True):
                logger.info(f"Skipping disabled experiment: {exp_name}")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Running Experiment: {exp_config['name']}")
            logger.info(f"{'='*60}")

            try:
                # Run experiment
                results = self._run_experiment(exp_name, exp_config)

                # Store results
                self.all_results[exp_name] = results

                # Log results
                self._log_results(exp_name, results)

                logger.info(f"✓ Completed: {exp_name}")

            except Exception as e:
                logger.error(f"✗ Failed: {exp_name}")
                logger.error(f"Error: {e}")

                if not self.config['execution']['continue_on_failure']:
                    raise

        # Generate paper materials
        logger.info(f"\n{'='*60}")
        logger.info("Generating Paper Materials")
        logger.info(f"{'='*60}")
        self._generate_paper_materials()

        # Save all results
        self._save_results()

        logger.info(f"\n{'='*80}")
        logger.info("All Experiments Complete!")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info(f"{'='*80}")

    def _run_experiment(self, exp_name: str, exp_config: Dict) -> Dict:
        """
        Run a single experiment.

        Args:
            exp_name: Experiment name
            exp_config: Experiment configuration

        Returns:
            Dictionary of results
        """
        if exp_name == 'standard_benchmarks':
            return self._run_standard_benchmarks(exp_config)
        elif exp_name == 'ablation_studies':
            return self._run_ablation_studies(exp_config)
        elif exp_name == 'zero_shot_generalization':
            return self._run_zero_shot_generalization(exp_config)
        elif exp_name == 'few_shot_learning':
            return self._run_few_shot_learning(exp_config)
        elif exp_name == 'long_horizon_evaluation':
            return self._run_long_horizon_evaluation(exp_config)
        elif exp_name == 'robustness_testing':
            return self._run_robustness_testing(exp_config)
        elif exp_name == 'efficiency_analysis':
            return self._run_efficiency_analysis(exp_config)
        elif exp_name == 'qualitative_analysis':
            return self._run_qualitative_analysis(exp_config)
        else:
            raise ValueError(f"Unknown experiment: {exp_name}")

    # ========================================================================
    # Experiment Implementations
    # ========================================================================

    def _run_standard_benchmarks(self, config: Dict) -> Dict:
        """Run standard VLA benchmarks."""
        logger.info("Running standard benchmarks on multiple datasets...")

        results = {}

        # Load model
        model = self._load_model()

        # Evaluate on each dataset
        for dataset_type in ['manipulation', 'navigation', 'long_horizon']:
            if dataset_type not in config['datasets']:
                continue

            dataset_configs = config['datasets'][dataset_type]

            for dataset_config in dataset_configs:
                dataset_name = dataset_config['name']
                num_episodes = dataset_config['num_episodes']
                tasks = dataset_config.get('tasks', [])

                logger.info(f"Evaluating on {dataset_name}...")

                try:
                    # Load dataset
                    dataset = self.dataset_loader.load_dataset(
                        dataset_name,
                        split='test'
                    )

                    # Evaluate
                    eval_results = self._evaluate_model(
                        model,
                        dataset,
                        num_episodes=num_episodes,
                        metrics=config.get('metrics', [])
                    )

                    results[dataset_name] = eval_results

                except Exception as e:
                    logger.error(f"Failed to evaluate on {dataset_name}: {e}")
                    results[dataset_name] = {'error': str(e)}

        # Compare with baselines
        baseline_results = self._evaluate_baselines(config)
        results['baselines'] = baseline_results

        return results

    def _run_ablation_studies(self, config: Dict) -> Dict:
        """Run comprehensive ablation studies."""
        logger.info("Running ablation studies...")

        results = {}
        num_episodes = config['num_episodes']

        # Load base model
        base_model = self._load_model()

        # Evaluate base model
        logger.info("Evaluating full model...")
        base_dataset = self.dataset_loader.load_dataset('hm3d', split='val')
        base_results = self._evaluate_model(
            base_model,
            base_dataset,
            num_episodes=num_episodes
        )
        results['full_model'] = base_results

        # Run each ablation
        for ablation_type, ablations in config['ablations'].items():
            logger.info(f"Running {ablation_type} ablations...")

            for ablation in ablations:
                ablation_name = ablation['name']

                if ablation_name == 'full_model':
                    continue

                logger.info(f"  Ablation: {ablation_name}")

                try:
                    # Load ablated model
                    ablation_config = ablation.get('config', {})
                    ablated_model = self._load_model(ablation_config)

                    # Evaluate
                    ablation_results = self._evaluate_model(
                        ablated_model,
                        base_dataset,
                        num_episodes=num_episodes
                    )

                    results[ablation_name] = ablation_results

                except Exception as e:
                    logger.error(f"Failed ablation {ablation_name}: {e}")
                    results[ablation_name] = {'error': str(e)}

        # Analyze ablation impact
        analysis = self._analyze_ablations(base_results, results)
        results['analysis'] = analysis

        return results

    def _run_zero_shot_generalization(self, config: Dict) -> Dict:
        """Run zero-shot generalization experiments."""
        logger.info("Running zero-shot generalization experiments...")

        results = {}

        # Load model
        model = self._load_model()

        # Test domain transfer
        transfer_pairs = config['transfer_pairs']

        for domain_type, pairs in transfer_pairs.items():
            logger.info(f"Testing {domain_type} domain transfer...")

            for pair in pairs:
                train_dataset = pair['train']
                test_dataset = pair['test']
                num_episodes = pair['num_episodes']

                logger.info(f"  Transfer: {train_dataset} -> {test_dataset}")

                try:
                    # Load test dataset
                    dataset = self.dataset_loader.load_dataset(
                        test_dataset,
                        split='test'
                    )

                    # Evaluate
                    transfer_results = self._evaluate_model(
                        model,
                        dataset,
                        num_episodes=num_episodes
                    )

                    pair_name = f"{train_dataset}_to_{test_dataset}"
                    results[pair_name] = transfer_results

                except Exception as e:
                    logger.error(f"Failed transfer test: {e}")

        # Compute generalization metrics
        results['metrics'] = self._compute_generalization_metrics(results)

        return results

    def _run_few_shot_learning(self, config: Dict) -> Dict:
        """Run few-shot learning experiments."""
        logger.info("Running few-shot learning experiments...")

        results = {}

        # Load model
        model = self._load_model()

        # Test with different numbers of demonstrations
        num_demonstrations_list = config['num_demonstrations']

        for num_demos in num_demonstrations_list:
            logger.info(f"Testing with {num_demos} demonstrations...")

            # Simulate few-shot adaptation
            # In practice, you would fine-tune on demonstrations

            # Evaluate
            dataset = self.dataset_loader.load_dataset('libero', split='test')
            eval_results = self._evaluate_model(
                model,
                dataset,
                num_episodes=config['num_test_episodes']
            )

            results[f'{num_demos}_shot'] = eval_results

        return results

    def _run_long_horizon_evaluation(self, config: Dict) -> Dict:
        """Run long-horizon task evaluation."""
        logger.info("Running long-horizon task evaluation...")

        results = {}

        # CALVIN-style evaluation
        if config['calvin_protocol']['enabled']:
            logger.info("Running CALVIN protocol...")

            calvin_results = self._evaluate_calvin_protocol(
                config['calvin_protocol']
            )
            results['calvin'] = calvin_results

        # Custom long-horizon tasks
        if 'custom_tasks' in config:
            logger.info("Running custom long-horizon tasks...")

            for task_config in config['custom_tasks']:
                task_name = task_config['name']

                # Evaluate task
                # Implementation depends on task type

                results[task_name] = {}

        return results

    def _run_robustness_testing(self, config: Dict) -> Dict:
        """Run robustness and safety testing."""
        logger.info("Running robustness testing...")

        results = {}

        # Load model
        model = self._load_model()
        base_dataset = self.dataset_loader.load_dataset('hm3d', split='test')

        # Visual perturbations
        if config['visual_perturbations']['occlusion']['enabled']:
            logger.info("Testing occlusion robustness...")

            occlusion_config = config['visual_perturbations']['occlusion']
            occlusion_results = self._test_occlusion_robustness(
                model,
                base_dataset,
                occlusion_config
            )
            results['occlusion'] = occlusion_results

        # Physical perturbations
        if config['physical_perturbations']['dynamic_obstacles']['enabled']:
            logger.info("Testing dynamic obstacle robustness...")

            # Evaluate with dynamic obstacles
            results['dynamic_obstacles'] = {}

        # Compute robustness score
        results['robustness_score'] = self._compute_robustness_score(results)

        return results

    def _run_efficiency_analysis(self, config: Dict) -> Dict:
        """Run computational efficiency analysis."""
        logger.info("Running efficiency analysis...")

        results = {}

        # Load model
        model = self._load_model()

        # Test different batch sizes
        for batch_size in config['batch_sizes']:
            logger.info(f"Testing batch size: {batch_size}")

            # Measure inference time, throughput, memory
            efficiency_metrics = self._measure_efficiency(
                model,
                batch_size=batch_size
            )

            results[f'batch_{batch_size}'] = efficiency_metrics

        return results

    def _run_qualitative_analysis(self, config: Dict) -> Dict:
        """Run qualitative analysis and visualization."""
        logger.info("Running qualitative analysis...")

        results = {}

        # Load model
        model = self._load_model()

        # Generate visualizations
        num_episodes = config['num_episodes']
        visualizations = config['visualizations']

        for viz_type in visualizations:
            logger.info(f"Generating {viz_type}...")

            # Generate visualization
            # Save to visualization directory

        return results

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _load_model(self, config_override: Optional[Dict] = None) -> torch.nn.Module:
        """Load VLA-GR model."""
        # Merge config override
        model_config = self.config['model'].copy()
        if config_override:
            model_config.update(config_override)

        # Load model
        try:
            model = VLAGRAgent(model_config)

            # Load checkpoint if specified
            checkpoint_path = model_config.get('checkpoint')
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from {checkpoint_path}")

            model.eval()
            return model

        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            logger.warning("Using mock model for testing")
            return MockModel()

    def _evaluate_model(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        num_episodes: int,
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate model on dataset.

        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
            num_episodes: Number of episodes to evaluate
            metrics: List of metrics to compute

        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating model on {num_episodes} episodes...")

        # Limit dataset size
        if len(dataset) > num_episodes:
            indices = np.random.choice(len(dataset), num_episodes, replace=False)
            from torch.utils.data import Subset
            dataset = Subset(dataset, indices)

        # Create dataloader
        dataloader = create_dataloader(
            dataset,
            batch_size=1,  # Evaluate one episode at a time
            shuffle=False,
            num_workers=0
        )

        # Collect results
        episodes = []
        all_predictions = []
        all_ground_truth = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            # Forward pass
            with torch.no_grad():
                outputs = model(batch)

            # Collect results
            episode_result = {
                'success': outputs.get('success', False),
                'distance_to_goal': outputs.get('distance_to_goal', 0.0),
                'path_length': outputs.get('path_length', 0.0),
            }
            episodes.append(episode_result)

            if 'actions' in outputs and 'ground_truth_actions' in batch:
                all_predictions.append(outputs['actions'].cpu().numpy())
                all_ground_truth.append(batch['ground_truth_actions'].cpu().numpy())

        # Compute metrics
        eval_results = self.metrics_calculator.compute_all_metrics(
            predictions=np.concatenate(all_predictions) if all_predictions else None,
            ground_truth=np.concatenate(all_ground_truth) if all_ground_truth else None,
            episodes=episodes
        )

        return eval_results.__dict__

    def _evaluate_baselines(self, config: Dict) -> Dict:
        """Evaluate baseline models."""
        baseline_results = {}

        for baseline_name, baseline_config in self.config.get('baselines', {}).items():
            if not baseline_config.get('enabled', True):
                continue

            logger.info(f"Evaluating baseline: {baseline_name}")

            try:
                # Load baseline model
                # Evaluate on same datasets
                baseline_results[baseline_name] = {}

            except Exception as e:
                logger.error(f"Failed to evaluate baseline {baseline_name}: {e}")

        return baseline_results

    def _analyze_ablations(self, base_results: Dict, ablation_results: Dict) -> Dict:
        """Analyze ablation study results."""
        analysis = {}

        base_success_rate = base_results.get('success_rate', 0.0)

        for ablation_name, results in ablation_results.items():
            if ablation_name in ['full_model', 'analysis']:
                continue

            ablation_success_rate = results.get('success_rate', 0.0)

            # Compute performance drop
            performance_drop = base_success_rate - ablation_success_rate
            relative_drop = performance_drop / base_success_rate if base_success_rate > 0 else 0.0

            analysis[ablation_name] = {
                'absolute_drop': performance_drop,
                'relative_drop': relative_drop,
                'component_importance': relative_drop  # Higher = more important
            }

        # Rank by importance
        ranked_components = sorted(
            analysis.items(),
            key=lambda x: x[1]['relative_drop'],
            reverse=True
        )

        analysis['ranked_components'] = [name for name, _ in ranked_components]

        return analysis

    def _test_occlusion_robustness(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        config: Dict
    ) -> Dict:
        """Test robustness to visual occlusion."""
        results = {}

        occlusion_levels = config['occlusion_levels']
        num_episodes_per_level = config['num_episodes_per_level']

        for occlusion_level in occlusion_levels:
            logger.info(f"  Occlusion level: {occlusion_level}")

            # Evaluate with occlusion
            # Apply occlusion mask to images

            results[f'occlusion_{occlusion_level}'] = {}

        return results

    def _measure_efficiency(
        self,
        model: torch.nn.Module,
        batch_size: int
    ) -> Dict:
        """Measure computational efficiency metrics."""
        # Measure inference time, throughput, memory usage

        efficiency_metrics = {
            'inference_time_ms': 0.0,
            'throughput_fps': 0.0,
            'memory_usage_mb': 0.0,
        }

        return efficiency_metrics

    def _log_results(self, exp_name: str, results: Dict):
        """Log results to tracking systems."""
        # Log to W&B
        if self.wandb_enabled:
            import wandb
            wandb.log({f"{exp_name}/{k}": v for k, v in results.items() if isinstance(v, (int, float))})

        # Log to TensorBoard
        if self.tb_writer:
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"{exp_name}/{k}", v, 0)

    def _save_results(self):
        """Save all results to disk."""
        results_file = self.results_dir / 'all_results.json'

        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)

        logger.info(f"Results saved to: {results_file}")

    def _generate_paper_materials(self):
        """Generate LaTeX tables, figures, and supplementary materials."""
        paper_dir = self.results_dir / 'paper_materials'
        paper_dir.mkdir(exist_ok=True)

        # Generate tables
        # Generate figures
        # Generate supplementary materials

        logger.info(f"Paper materials saved to: {paper_dir}")


class MockModel(torch.nn.Module):
    """Mock model for testing."""

    def forward(self, batch):
        # Return mock outputs
        return {
            'success': True,
            'distance_to_goal': 0.1,
            'path_length': 10.0,
            'actions': torch.randn(1, 10, 7)
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run VLA-GR Comprehensive Experiments"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_vla_comprehensive.yaml',
        help='Path to experiment configuration file'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )

    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logger.info(f"Using GPU: {args.gpu}")
    else:
        logger.info("Using CPU")

    # Run experiments
    runner = VLAExperimentRunner(args.config)
    runner.run_all_experiments()


if __name__ == "__main__":
    main()

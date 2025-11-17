"""
VLA Standard Evaluation Metrics
Based on ICLR 2025, ICML 2024, and latest VLA research
References: arXiv 2411.05821 (VLA Benchmarking), OpenVLA, GPT-4o
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)


@dataclass
class VLAMetricResults:
    """Container for VLA evaluation metric results."""

    # Action prediction metrics
    mse: Optional[float] = None
    rmse: Optional[float] = None
    amse: Optional[float] = None  # Average MSE across trajectories
    namse: Optional[float] = None  # Normalized AMSE
    action_accuracy: Optional[float] = None  # For discrete actions

    # Task completion metrics
    success_rate: Optional[float] = None
    completion_rate: Optional[float] = None
    partial_completion_rate: Optional[float] = None

    # Sequence metrics (for multi-step tasks)
    avg_sequence_length: Optional[float] = None
    success_rate_1: Optional[float] = None  # CALVIN-style
    success_rate_2: Optional[float] = None
    success_rate_3: Optional[float] = None
    success_rate_4: Optional[float] = None
    success_rate_5: Optional[float] = None

    # Efficiency metrics
    spl: Optional[float] = None  # Success weighted by Path Length
    soft_spl: Optional[float] = None
    task_completion_time: Optional[float] = None
    path_efficiency: Optional[float] = None

    # Safety metrics
    collision_rate: Optional[float] = None
    safety_violation_rate: Optional[float] = None

    # Manipulation-specific
    grasp_success_rate: Optional[float] = None
    placement_accuracy: Optional[float] = None
    manipulation_accuracy: Optional[float] = None

    # Navigation-specific
    distance_to_goal: Optional[float] = None
    navigation_error: Optional[float] = None

    # Robustness metrics
    zero_shot_performance: Optional[float] = None
    generalization_gap: Optional[float] = None

    # Statistical measures
    confidence_interval: Optional[Tuple[float, float]] = None
    std_dev: Optional[float] = None

    # Additional metadata
    num_episodes: int = 0
    metadata: Dict = field(default_factory=dict)


class VLAMetricsCalculator:
    """
    Calculator for VLA standard evaluation metrics.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize metrics calculator.

        Args:
            config: Configuration dictionary for metrics computation
        """
        self.config = config or {}
        self.reset()

    def reset(self):
        """Reset internal state for new evaluation."""
        self.predictions = []
        self.ground_truth = []
        self.episodes = []
        self.trajectories = []

    # ========================================================================
    # Action Prediction Metrics
    # ========================================================================

    def compute_mse(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        per_dimension: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Compute Mean Squared Error.

        Args:
            predictions: Predicted actions [N, action_dim]
            ground_truth: Ground truth actions [N, action_dim]
            per_dimension: If True, return MSE per dimension

        Returns:
            MSE value or array of per-dimension MSEs
        """
        squared_errors = (predictions - ground_truth) ** 2

        if per_dimension:
            return np.mean(squared_errors, axis=0)
        else:
            return np.mean(squared_errors)

    def compute_rmse(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """
        Compute Root Mean Squared Error.

        Args:
            predictions: Predicted actions
            ground_truth: Ground truth actions

        Returns:
            RMSE value
        """
        mse = self.compute_mse(predictions, ground_truth)
        return np.sqrt(mse)

    def compute_amse(
        self,
        trajectory_predictions: List[np.ndarray],
        trajectory_ground_truth: List[np.ndarray]
    ) -> float:
        """
        Compute Average Mean Squared Error across trajectories.
        This is the standard VLA benchmark metric (arXiv 2411.05821).

        Args:
            trajectory_predictions: List of predicted action sequences
            trajectory_ground_truth: List of ground truth action sequences

        Returns:
            AMSE value
        """
        mses = []

        for pred_traj, gt_traj in zip(trajectory_predictions, trajectory_ground_truth):
            # Compute MSE for this trajectory
            traj_mse = self.compute_mse(pred_traj, gt_traj)
            mses.append(traj_mse)

        # Average across trajectories
        return np.mean(mses)

    def compute_namse(
        self,
        trajectory_predictions: List[np.ndarray],
        trajectory_ground_truth: List[np.ndarray],
        normalization_method: str = "prediction_range"
    ) -> float:
        """
        Compute Normalized Average Mean Squared Error.
        Normalized by action prediction range for fair cross-model comparison.

        Args:
            trajectory_predictions: List of predicted action sequences
            trajectory_ground_truth: List of ground truth action sequences
            normalization_method: "prediction_range", "ground_truth_range", or "action_space_range"

        Returns:
            NAMSE value
        """
        # Compute AMSE
        amse = self.compute_amse(trajectory_predictions, trajectory_ground_truth)

        # Compute normalization factor
        all_predictions = np.concatenate(trajectory_predictions, axis=0)

        if normalization_method == "prediction_range":
            pred_range = np.ptp(all_predictions, axis=0)  # max - min per dimension
            normalization = np.mean(pred_range ** 2)

        elif normalization_method == "ground_truth_range":
            all_ground_truth = np.concatenate(trajectory_ground_truth, axis=0)
            gt_range = np.ptp(all_ground_truth, axis=0)
            normalization = np.mean(gt_range ** 2)

        elif normalization_method == "action_space_range":
            # Assume action space is [-1, 1]
            normalization = 4.0  # (1 - (-1))^2

        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")

        # Avoid division by zero
        if normalization == 0:
            logger.warning("Normalization factor is zero, returning AMSE")
            return amse

        return amse / normalization

    def compute_action_accuracy(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        top_k: int = 1
    ) -> float:
        """
        Compute action accuracy for discrete action spaces.

        Args:
            predictions: Predicted action probabilities or indices
            ground_truth: Ground truth action indices
            top_k: Consider top-k predictions

        Returns:
            Accuracy value
        """
        if top_k == 1:
            # Top-1 accuracy
            pred_actions = np.argmax(predictions, axis=-1) if len(predictions.shape) > 1 else predictions
            accuracy = np.mean(pred_actions == ground_truth)
        else:
            # Top-k accuracy
            top_k_preds = np.argsort(predictions, axis=-1)[:, -top_k:]
            accuracy = np.mean([gt in top_k_preds[i] for i, gt in enumerate(ground_truth)])

        return accuracy

    # ========================================================================
    # Task Completion Metrics
    # ========================================================================

    def compute_success_rate(
        self,
        successes: List[bool],
        confidence_level: float = 0.95
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Compute success rate with confidence interval.

        Args:
            successes: List of success indicators
            confidence_level: Confidence level for interval

        Returns:
            (success_rate, (lower_bound, upper_bound))
        """
        success_rate = np.mean(successes)

        # Compute confidence interval using bootstrap
        ci = self._bootstrap_confidence_interval(
            successes,
            statistic=np.mean,
            confidence_level=confidence_level
        )

        return success_rate, ci

    def compute_completion_rate(
        self,
        completions: List[float]
    ) -> float:
        """
        Compute task completion rate.
        For multi-step tasks, this can be partial completion.

        Args:
            completions: List of completion values (0.0 to 1.0)

        Returns:
            Average completion rate
        """
        return np.mean(completions)

    # ========================================================================
    # Sequence Metrics (CALVIN-style)
    # ========================================================================

    def compute_sequence_metrics(
        self,
        sequence_lengths: List[int],
        max_length: int = 5
    ) -> Dict[str, float]:
        """
        Compute CALVIN-style sequence metrics.

        Args:
            sequence_lengths: List of successfully completed sequence lengths
            max_length: Maximum sequence length to evaluate

        Returns:
            Dictionary of sequence metrics
        """
        metrics = {
            'avg_sequence_length': np.mean(sequence_lengths),
        }

        # Compute success rate for each sequence length
        for length in range(1, max_length + 1):
            success_count = sum(1 for seq_len in sequence_lengths if seq_len >= length)
            success_rate = success_count / len(sequence_lengths)
            metrics[f'success_rate_{length}'] = success_rate

        return metrics

    # ========================================================================
    # Efficiency Metrics
    # ========================================================================

    def compute_spl(
        self,
        successes: List[bool],
        path_lengths: List[float],
        optimal_path_lengths: List[float]
    ) -> float:
        """
        Compute Success weighted by Path Length (SPL).

        Args:
            successes: List of success indicators
            path_lengths: List of actual path lengths
            optimal_path_lengths: List of optimal (shortest) path lengths

        Returns:
            SPL value
        """
        spl_values = []

        for success, actual_length, optimal_length in zip(successes, path_lengths, optimal_path_lengths):
            if success:
                # Clip ratio to avoid values > 1
                ratio = min(1.0, optimal_length / actual_length) if actual_length > 0 else 0.0
                spl_values.append(ratio)
            else:
                spl_values.append(0.0)

        return np.mean(spl_values)

    def compute_soft_spl(
        self,
        distances_to_goal: List[float],
        path_lengths: List[float],
        optimal_path_lengths: List[float],
        success_threshold: float = 0.2
    ) -> float:
        """
        Compute Soft SPL with continuous success based on distance to goal.

        Args:
            distances_to_goal: Final distances to goal
            path_lengths: Actual path lengths
            optimal_path_lengths: Optimal path lengths
            success_threshold: Distance threshold for full success

        Returns:
            Soft SPL value
        """
        soft_spl_values = []

        for dist, actual_length, optimal_length in zip(distances_to_goal, path_lengths, optimal_path_lengths):
            # Soft success: exponential decay based on distance
            soft_success = np.exp(-2.0 * dist / success_threshold)

            # Path efficiency
            path_efficiency = min(1.0, optimal_length / actual_length) if actual_length > 0 else 0.0

            soft_spl_values.append(soft_success * path_efficiency)

        return np.mean(soft_spl_values)

    # ========================================================================
    # Manipulation-Specific Metrics
    # ========================================================================

    def compute_grasp_success_rate(
        self,
        grasp_successes: List[bool]
    ) -> float:
        """Compute grasp success rate."""
        return np.mean(grasp_successes)

    def compute_placement_accuracy(
        self,
        final_positions: np.ndarray,
        target_positions: np.ndarray,
        threshold: float = 0.05
    ) -> float:
        """
        Compute placement accuracy.

        Args:
            final_positions: Final object positions [N, 3]
            target_positions: Target positions [N, 3]
            threshold: Distance threshold for success (meters)

        Returns:
            Placement accuracy
        """
        distances = np.linalg.norm(final_positions - target_positions, axis=1)
        return np.mean(distances < threshold)

    # ========================================================================
    # Robustness Metrics
    # ========================================================================

    def compute_generalization_gap(
        self,
        train_performance: float,
        test_performance: float
    ) -> float:
        """
        Compute generalization gap.

        Args:
            train_performance: Performance on training set
            test_performance: Performance on test set

        Returns:
            Generalization gap (higher = worse generalization)
        """
        return train_performance - test_performance

    def compute_robustness_score(
        self,
        performance_vs_perturbation: List[Tuple[float, float]]
    ) -> float:
        """
        Compute robustness score as area under the performance curve.

        Args:
            performance_vs_perturbation: List of (perturbation_level, performance) tuples

        Returns:
            Robustness score (area under curve)
        """
        # Sort by perturbation level
        sorted_data = sorted(performance_vs_perturbation)
        perturbations = [x[0] for x in sorted_data]
        performances = [x[1] for x in sorted_data]

        # Compute area under curve using trapezoidal rule
        auc = np.trapz(performances, perturbations)

        # Normalize by perturbation range
        perturbation_range = max(perturbations) - min(perturbations)
        if perturbation_range > 0:
            auc /= perturbation_range

        return auc

    # ========================================================================
    # Statistical Utilities
    # ========================================================================

    def _bootstrap_confidence_interval(
        self,
        data: List,
        statistic: callable = np.mean,
        confidence_level: float = 0.95,
        num_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval.

        Args:
            data: Data to bootstrap
            statistic: Statistic to compute
            confidence_level: Confidence level
            num_bootstrap: Number of bootstrap samples

        Returns:
            (lower_bound, upper_bound)
        """
        data = np.array(data)
        bootstrap_samples = []

        for _ in range(num_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_samples.append(statistic(sample))

        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_samples, alpha / 2 * 100)
        upper = np.percentile(bootstrap_samples, (1 - alpha / 2) * 100)

        return (lower, upper)

    def statistical_significance_test(
        self,
        results_a: List[float],
        results_b: List[float],
        test_type: str = "t_test",
        significance_level: float = 0.05
    ) -> Tuple[bool, float]:
        """
        Test statistical significance between two sets of results.

        Args:
            results_a: Results from method A
            results_b: Results from method B
            test_type: "t_test", "wilcoxon", or "mann_whitney"
            significance_level: Significance level (alpha)

        Returns:
            (is_significant, p_value)
        """
        if test_type == "t_test":
            statistic, p_value = stats.ttest_ind(results_a, results_b)
        elif test_type == "wilcoxon":
            statistic, p_value = stats.wilcoxon(results_a, results_b)
        elif test_type == "mann_whitney":
            statistic, p_value = stats.mannwhitneyu(results_a, results_b)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        is_significant = p_value < significance_level

        return is_significant, p_value

    # ========================================================================
    # Comprehensive Evaluation
    # ========================================================================

    def compute_all_metrics(
        self,
        predictions: Optional[np.ndarray] = None,
        ground_truth: Optional[np.ndarray] = None,
        episodes: Optional[List[Dict]] = None,
        task_type: str = "manipulation"
    ) -> VLAMetricResults:
        """
        Compute all applicable metrics for a set of evaluation results.

        Args:
            predictions: Predicted actions
            ground_truth: Ground truth actions
            episodes: List of episode dictionaries with results
            task_type: "manipulation", "navigation", or "mobile_manipulation"

        Returns:
            VLAMetricResults object with all computed metrics
        """
        results = VLAMetricResults(num_episodes=len(episodes) if episodes else 0)

        # Action prediction metrics
        if predictions is not None and ground_truth is not None:
            results.mse = self.compute_mse(predictions, ground_truth)
            results.rmse = self.compute_rmse(predictions, ground_truth)

        # Episode-based metrics
        if episodes:
            successes = [ep.get('success', False) for ep in episodes]
            results.success_rate, results.confidence_interval = self.compute_success_rate(successes)

            # Task-specific metrics
            if task_type == "manipulation":
                if all('grasp_success' in ep for ep in episodes):
                    grasp_successes = [ep['grasp_success'] for ep in episodes]
                    results.grasp_success_rate = self.compute_grasp_success_rate(grasp_successes)

            elif task_type == "navigation":
                if all('distance_to_goal' in ep for ep in episodes):
                    distances = [ep['distance_to_goal'] for ep in episodes]
                    results.distance_to_goal = np.mean(distances)

                if all('path_length' in ep and 'optimal_path_length' in ep for ep in episodes):
                    path_lengths = [ep['path_length'] for ep in episodes]
                    optimal_lengths = [ep['optimal_path_length'] for ep in episodes]
                    results.spl = self.compute_spl(successes, path_lengths, optimal_lengths)

        return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create metrics calculator
    calculator = VLAMetricsCalculator()

    # Example: Compute action prediction metrics
    predictions = np.random.randn(100, 7)  # 100 timesteps, 7D actions
    ground_truth = np.random.randn(100, 7)

    mse = calculator.compute_mse(predictions, ground_truth)
    rmse = calculator.compute_rmse(predictions, ground_truth)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Example: Compute AMSE across trajectories
    traj_preds = [np.random.randn(50, 7) for _ in range(10)]
    traj_gt = [np.random.randn(50, 7) for _ in range(10)]

    amse = calculator.compute_amse(traj_preds, traj_gt)
    namse = calculator.compute_namse(traj_preds, traj_gt)

    print(f"AMSE: {amse:.4f}")
    print(f"NAMSE: {namse:.4f}")

    # Example: Compute success rate
    successes = [True] * 55 + [False] * 45
    success_rate, ci = calculator.compute_success_rate(successes)

    print(f"Success Rate: {success_rate:.2%} (95% CI: [{ci[0]:.2%}, {ci[1]:.2%}])")

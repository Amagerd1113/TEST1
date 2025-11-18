"""
VLN metrics computation including distractor-specific analysis.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cdist
from dtaidistance import dtw


def compute_vln_metrics(
    predicted_paths: List[np.ndarray],
    ground_truth_paths: List[np.ndarray],
    goal_positions: List[np.ndarray],
    success_threshold: float = 3.0,
) -> Dict[str, float]:
    """
    Compute standard VLN metrics: SR, SPL, Oracle Success, Path Length.

    Args:
        predicted_paths: List of (T, 3) predicted trajectories
        ground_truth_paths: List of (T, 3) ground truth trajectories
        goal_positions: List of (3,) goal positions
        success_threshold: Distance threshold for success (meters)

    Returns:
        metrics: Dictionary of metric values
    """
    num_episodes = len(predicted_paths)

    successes = []
    spls = []
    oracle_successes = []
    path_lengths = []

    for pred_path, gt_path, goal in zip(predicted_paths, ground_truth_paths, goal_positions):
        # Success: final position within threshold of goal
        final_pos = pred_path[-1]
        dist_to_goal = np.linalg.norm(final_pos - goal)
        success = dist_to_goal < success_threshold
        successes.append(success)

        # Oracle Success: any point on path within threshold
        min_dist_to_goal = np.min(np.linalg.norm(pred_path - goal, axis=1))
        oracle_success = min_dist_to_goal < success_threshold
        oracle_successes.append(oracle_success)

        # Path Length
        pred_length = compute_path_length(pred_path)
        gt_length = compute_path_length(gt_path)
        path_lengths.append(pred_length)

        # SPL: Success weighted by Path Length
        if success:
            spl = gt_length / max(pred_length, gt_length)
        else:
            spl = 0.0
        spls.append(spl)

    return {
        "success_rate": np.mean(successes),
        "spl": np.mean(spls),
        "oracle_success": np.mean(oracle_successes),
        "path_length": np.mean(path_lengths),
        "num_episodes": num_episodes,
    }


def compute_distractor_metrics(
    predicted_paths: List[np.ndarray],
    ground_truth_paths: List[np.ndarray],
    goal_positions: List[np.ndarray],
    distractor_positions: List[List[np.ndarray]],
    success_threshold: float = 3.0,
    similarity_threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Compute metrics specifically for high-distractor episodes.

    Args:
        predicted_paths: List of predicted trajectories
        ground_truth_paths: List of ground truth trajectories
        goal_positions: List of goal positions
        distractor_positions: List of lists of distractor positions per episode
        success_threshold: Distance for success
        similarity_threshold: Threshold to consider episode "high distractor"

    Returns:
        distractor_metrics: Metrics for high-distractor subset
    """
    # Filter episodes with high visual similarity distractors
    high_distractor_indices = []

    for i, distractors in enumerate(distractor_positions):
        if len(distractors) > 0:
            # Check if any distractor is close to ground truth path
            gt_path = ground_truth_paths[i]
            min_dist_to_path = float('inf')

            for distractor in distractors:
                dists = np.linalg.norm(gt_path - distractor, axis=1)
                min_dist_to_path = min(min_dist_to_path, np.min(dists))

            # If distractor is within 5m of path, consider high similarity
            if min_dist_to_path < 5.0:
                high_distractor_indices.append(i)

    if len(high_distractor_indices) == 0:
        return {
            "success_rate": 0.0,
            "spl": 0.0,
            "num_episodes": 0,
            "distractor_avoidance_rate": 0.0,
        }

    # Compute metrics on high-distractor subset
    pred_subset = [predicted_paths[i] for i in high_distractor_indices]
    gt_subset = [ground_truth_paths[i] for i in high_distractor_indices]
    goal_subset = [goal_positions[i] for i in high_distractor_indices]

    subset_metrics = compute_vln_metrics(pred_subset, gt_subset, goal_subset, success_threshold)

    # Distractor avoidance rate: fraction of episodes that didn't end at distractor
    avoidance_count = 0
    for i in high_distractor_indices:
        final_pos = predicted_paths[i][-1]
        at_distractor = False

        for distractor in distractor_positions[i]:
            if np.linalg.norm(final_pos - distractor) < success_threshold:
                at_distractor = True
                break

        if not at_distractor:
            avoidance_count += 1

    subset_metrics["distractor_avoidance_rate"] = avoidance_count / len(high_distractor_indices)

    return subset_metrics


def compute_path_length(path: np.ndarray) -> float:
    """
    Compute total path length.

    Args:
        path: (T, 3) positions

    Returns:
        length: Total Euclidean path length
    """
    if len(path) < 2:
        return 0.0

    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(segment_lengths)


def compute_ndtw(
    predicted_path: np.ndarray,
    ground_truth_path: np.ndarray,
) -> float:
    """
    Compute Normalized Dynamic Time Warping distance.

    Args:
        predicted_path: (T1, 3) predicted trajectory
        ground_truth_path: (T2, 3) ground truth trajectory

    Returns:
        ndtw: Normalized DTW distance
    """
    try:
        distance = dtw.distance(predicted_path, ground_truth_path)
        # Normalize by path lengths
        ndtw = distance / (len(predicted_path) + len(ground_truth_path))
        return ndtw
    except:
        # Fallback to Euclidean if DTW fails
        return compute_path_length(predicted_path - ground_truth_path)


def analyze_slingshot_behavior(
    trajectory: np.ndarray,
    distractor_positions: List[np.ndarray],
    goal_position: np.ndarray,
) -> Dict[str, float]:
    """
    Analyze if trajectory exhibits true slingshot behavior.

    Metrics:
    - Curvature near distractors (should be high)
    - Minimum distance to distractors (should be large)
    - Final approach angle (should align with goal)

    Args:
        trajectory: (T, 3) trajectory
        distractor_positions: List of (3,) distractor positions
        goal_position: (3,) goal position

    Returns:
        analysis: Dictionary of slingshot metrics
    """
    if len(trajectory) < 3:
        return {"curvature": 0.0, "min_distractor_distance": 0.0, "final_angle_error": 0.0}

    # Compute path curvature
    velocities = np.diff(trajectory, axis=0)
    accelerations = np.diff(velocities, axis=0)
    curvatures = np.linalg.norm(accelerations, axis=1) / (np.linalg.norm(velocities[:-1], axis=1) + 1e-6)
    mean_curvature = np.mean(curvatures)

    # Minimum distance to any distractor
    if len(distractor_positions) > 0:
        dists_to_distractors = []
        for distractor in distractor_positions:
            dists = np.linalg.norm(trajectory - distractor, axis=1)
            dists_to_distractors.append(np.min(dists))
        min_distractor_distance = np.min(dists_to_distractors)
    else:
        min_distractor_distance = float('inf')

    # Final approach angle to goal
    final_velocity = velocities[-1]
    goal_direction = goal_position - trajectory[-1]
    final_velocity_norm = final_velocity / (np.linalg.norm(final_velocity) + 1e-6)
    goal_direction_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)
    cos_angle = np.dot(final_velocity_norm, goal_direction_norm)
    angle_error = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

    return {
        "curvature": mean_curvature,
        "min_distractor_distance": min_distractor_distance,
        "final_angle_error": angle_error,
    }

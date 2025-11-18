"""
Visualization utilities for scalar fields, metrics, and trajectories.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import imageio
from typing import Optional, List


def visualize_scalar_fields(
    phi_positive: np.ndarray,
    phi_negative: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Visualize dual scalar fields φ+ and φ- as 2D slices.

    Args:
        phi_positive: (H, W, D) attractive field
        phi_negative: (H, W, D) repulsive field
        save_path: Path to save figure
    """
    H, W, D = phi_positive.shape
    mid_slice = D // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # φ+ (attractive)
    im1 = axes[0].imshow(phi_positive[:, :, mid_slice], cmap='Reds', origin='lower')
    axes[0].set_title('Attractive Field φ₊\n(Target)', fontsize=14)
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('X')
    plt.colorbar(im1, ax=axes[0], label='Field Strength')

    # φ- (repulsive)
    im2 = axes[1].imshow(phi_negative[:, :, mid_slice], cmap='Blues', origin='lower')
    axes[1].set_title('Repulsive Field φ₋\n(Distractors)', fontsize=14)
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('X')
    plt.colorbar(im2, ax=axes[1], label='Field Strength')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def visualize_geodesic(
    trajectory: np.ndarray,
    Phi: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Visualize geodesic trajectory on conformal factor field.

    Args:
        trajectory: (num_steps, 3) trajectory positions
        Phi: (H, W, D) conformal factor
        save_path: Path to save figure
    """
    H, W, D = Phi.shape
    mid_slice = D // 2

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot conformal factor as background
    im = ax.imshow(Phi[:, :, mid_slice], cmap='viridis', origin='lower', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Conformal Factor Φ')

    # Plot trajectory
    traj_x = trajectory[:, 0] * H
    traj_y = trajectory[:, 1] * W
    ax.plot(traj_y, traj_x, 'r-', linewidth=2, label='Geodesic')
    ax.plot(traj_y[0], traj_x[0], 'go', markersize=10, label='Start')
    ax.plot(traj_y[-1], traj_x[-1], 'r*', markersize=15, label='End')

    ax.set_title('Gravitational Slingshot Trajectory', fontsize=16)
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_trajectory(
    positions: List[np.ndarray],
    labels: List[str],
    save_path: Optional[str] = None,
):
    """
    Plot multiple trajectories for comparison.

    Args:
        positions: List of (N, 2) position arrays
        labels: List of trajectory labels
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for pos, label, color in zip(positions, labels, colors):
        ax.plot(pos[:, 0], pos[:, 1], '-', color=color, linewidth=2, label=label, alpha=0.7)
        ax.plot(pos[0, 0], pos[0, 1], 'o', color=color, markersize=8)
        ax.plot(pos[-1, 0], pos[-1, 1], '*', color=color, markersize=12)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Navigation Trajectories', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def create_slingshot_animation(
    trajectory: np.ndarray,
    Phi: np.ndarray,
    save_path: str,
    fps: int = 10,
):
    """
    Create animated visualization of slingshot trajectory.

    Args:
        trajectory: (T, 3) positions over time
        Phi: (H, W, D) conformal factor
        save_path: Path to save .gif or .mp4
        fps: Frames per second
    """
    H, W, D = Phi.shape
    mid_slice = D // 2

    frames = []

    for t in range(len(trajectory)):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Background field
        ax.imshow(Phi[:, :, mid_slice], cmap='viridis', origin='lower', alpha=0.6)

        # Trajectory up to time t
        traj_x = trajectory[:t+1, 0] * H
        traj_y = trajectory[:t+1, 1] * W
        ax.plot(traj_y, traj_x, 'r-', linewidth=2)
        ax.plot(traj_y[-1], traj_x[-1], 'ro', markersize=10)

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_title(f'Slingshot Navigation (t={t})', fontsize=14)
        ax.axis('off')

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.close()

    # Save animation
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"✓ Animation saved to {save_path}")

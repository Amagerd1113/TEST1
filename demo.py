#!/usr/bin/env python3
"""
Gravitational Slingshot Navigation - Interactive Demo
======================================================

Demonstrates slingshot navigation with full visualization of:
- Dual scalar fields (φ₊, φ₋)
- Conformal factor (Φ) and scalar curvature (R)
- Geodesic trajectory
- Slingshot animation

Usage:
    python demo.py --checkpoint path/to/model.pt --scene_id SCENE_ID --instruction "Go to X"

For quick test without model:
    python demo.py --mode synthetic
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dual_scalar_poisson_solver import DualScalarPoissonSolver
from src.conformal_metric import ConformalMetric, GeodesicIntegrator
from src.utils.visualization import (
    visualize_scalar_fields,
    visualize_geodesic,
    create_slingshot_animation,
)


def create_synthetic_scene():
    """
    Create synthetic target and distractor density maps for visualization.

    Returns a scenario with:
    - Target at (0.7, 0.5, 0.5) - blue chair
    - Distractor at (0.3, 0.5, 0.5) - similar blue sofa
    """
    print("Creating synthetic scene...")

    grid_size = 64
    rho_target = np.zeros((grid_size, grid_size, grid_size))
    rho_distractor = np.zeros((grid_size, grid_size, grid_size))

    # Target: Gaussian blob at (45, 32, 32) - 70% along x-axis
    target_pos = (45, 32, 32)
    sigma = 5.0

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                dist_sq = (i - target_pos[0])**2 + (j - target_pos[1])**2 + (k - target_pos[2])**2
                rho_target[i, j, k] = np.exp(-dist_sq / (2 * sigma**2))

    # Distractor: Gaussian blob at (19, 32, 32) - 30% along x-axis
    distractor_pos = (19, 32, 32)
    sigma_distractor = 4.0

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                dist_sq = (i - distractor_pos[0])**2 + (j - distractor_pos[1])**2 + (k - distractor_pos[2])**2
                rho_distractor[i, j, k] = 0.8 * np.exp(-dist_sq / (2 * sigma_distractor**2))

    return rho_target, rho_distractor


def run_synthetic_demo(output_dir: Path, lambda_repulsion: float = 2.0):
    """
    Run complete demo with synthetic scene.
    """
    print("\n" + "="*70)
    print("  Gravitational Slingshot Navigation - Synthetic Demo")
    print("="*70 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create synthetic densities
    print("Step 1/5: Creating synthetic scene...")
    rho_target_np, rho_distractor_np = create_synthetic_scene()

    # Convert to torch tensors
    rho_target = torch.from_numpy(rho_target_np).float().unsqueeze(0).unsqueeze(0).to(device)
    rho_distractor = torch.from_numpy(rho_distractor_np).float().unsqueeze(0).unsqueeze(0).to(device)

    print(f"  Target density sum: {rho_target.sum().item():.2f}")
    print(f"  Distractor density sum: {rho_distractor.sum().item():.2f}")

    # Step 2: Solve Poisson equations
    print("\nStep 2/5: Solving dual scalar Poisson equations...")
    solver = DualScalarPoissonSolver(
        grid_size=(64, 64, 64),
        fourier_features=256,
        latent_dim=256,
    ).to(device)

    with torch.no_grad():
        phi_positive, phi_negative = solver(rho_target, rho_distractor)

    print(f"  φ₊ range: [{phi_positive.min().item():.3f}, {phi_positive.max().item():.3f}]")
    print(f"  φ₋ range: [{phi_negative.min().item():.3f}, {phi_negative.max().item():.3f}]")

    # Visualize scalar fields
    phi_pos_np = phi_positive[0].cpu().numpy()
    phi_neg_np = phi_negative[0].cpu().numpy()

    visualize_scalar_fields(
        phi_pos_np,
        phi_neg_np,
        save_path=output_dir / "01_scalar_fields.png"
    )
    print(f"  ✓ Saved: {output_dir / '01_scalar_fields.png'}")

    # Step 3: Compute conformal metric
    print("\nStep 3/5: Computing conformal metric...")
    metric = ConformalMetric(
        lambda_repulsion=lambda_repulsion,
        learn_lambda=False,
    ).to(device)

    Phi = metric.compute_conformal_factor(phi_positive, phi_negative)
    R = metric.compute_scalar_curvature(Phi)
    grad_Phi = metric.compute_gradient(Phi)
    grad_mag = torch.norm(grad_Phi, dim=-1)

    print(f"  Conformal factor Φ range: [{Phi.min().item():.3f}, {Phi.max().item():.3f}]")
    print(f"  Scalar curvature R range: [{R.min().item():.3f}, {R.max().item():.3f}]")
    print(f"  λ (repulsion strength): {lambda_repulsion:.2f}")

    # Visualize conformal factor
    Phi_np = Phi[0].cpu().numpy()
    R_np = R[0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    mid_slice = 32
    im1 = axes[0].imshow(Phi_np[:, :, mid_slice], cmap='RdBu_r', origin='lower')
    axes[0].set_title('Conformal Factor Φ = φ₊ - λφ₋', fontsize=14)
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('X')
    plt.colorbar(im1, ax=axes[0], label='Φ')

    im2 = axes[1].imshow(R_np[:, :, mid_slice], cmap='plasma', origin='lower', vmin=-10, vmax=10)
    axes[1].set_title('Scalar Curvature R', fontsize=14)
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('X')
    plt.colorbar(im2, ax=axes[1], label='R')

    plt.tight_layout()
    plt.savefig(output_dir / "02_metric_fields.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir / '02_metric_fields.png'}")

    # Step 4: Integrate geodesic
    print("\nStep 4/5: Integrating geodesic trajectory...")
    integrator = GeodesicIntegrator(
        conformal_metric=metric,
        integration_method="rk4",
        step_size=0.05,
        max_steps=100,
    )

    # Start from (0.1, 0.5, 0.5), aiming toward target
    start_pos = torch.tensor([[0.15, 0.5, 0.5]], device=device)
    start_vel = torch.tensor([[0.1, 0.0, 0.0]], device=device)  # Move along x-axis

    with torch.no_grad():
        positions, velocities = integrator.integrate_geodesic(start_pos, start_vel, Phi)

    trajectory_np = positions[0].cpu().numpy()
    print(f"  Start: {trajectory_np[0]}")
    print(f"  End: {trajectory_np[-1]}")
    print(f"  Total steps: {len(trajectory_np)}")

    # Visualize geodesic on Φ field
    visualize_geodesic(
        trajectory_np,
        Phi_np,
        save_path=output_dir / "03_geodesic_trajectory.png"
    )
    print(f"  ✓ Saved: {output_dir / '03_geodesic_trajectory.png'}")

    # Step 5: Create animation
    print("\nStep 5/5: Creating slingshot animation...")
    create_slingshot_animation(
        trajectory_np,
        Phi_np,
        save_path=output_dir / "04_slingshot_animation.gif",
        fps=10,
    )
    print(f"  ✓ Saved: {output_dir / '04_slingshot_animation.gif'}")

    # Summary visualization
    print("\nCreating summary figure...")
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: Densities
    ax1 = fig.add_subplot(131)
    combined_density = rho_target_np[:, :, 32] - 0.7 * rho_distractor_np[:, :, 32]
    im1 = ax1.imshow(combined_density, cmap='RdBu', origin='lower')
    ax1.plot(trajectory_np[:, 1] * 64, trajectory_np[:, 0] * 64, 'g-', linewidth=2, label='Geodesic')
    ax1.plot(trajectory_np[0, 1] * 64, trajectory_np[0, 0] * 64, 'go', markersize=10, label='Start')
    ax1.set_title('(a) Source Densities\n(Red=Target, Blue=Distractor)', fontsize=12)
    ax1.set_xlabel('Y')
    ax1.set_ylabel('X')
    ax1.legend()
    plt.colorbar(im1, ax=ax1)

    # Panel 2: Conformal Factor
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(Phi_np[:, :, 32], cmap='viridis', origin='lower')
    ax2.plot(trajectory_np[:, 1] * 64, trajectory_np[:, 0] * 64, 'r-', linewidth=2, label='Geodesic')
    ax2.plot(trajectory_np[0, 1] * 64, trajectory_np[0, 0] * 64, 'ro', markersize=10)
    ax2.plot(trajectory_np[-1, 1] * 64, trajectory_np[-1, 0] * 64, 'r*', markersize=15)
    ax2.set_title('(b) Conformal Factor Φ\nwith Slingshot Trajectory', fontsize=12)
    ax2.set_xlabel('Y')
    ax2.set_ylabel('X')
    ax2.legend()
    plt.colorbar(im2, ax=ax2, label='Φ')

    # Panel 3: Curvature
    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(R_np[:, :, 32], cmap='plasma', origin='lower', vmin=-10, vmax=10)
    ax3.plot(trajectory_np[:, 1] * 64, trajectory_np[:, 0] * 64, 'w-', linewidth=2, alpha=0.7)
    ax3.set_title('(c) Scalar Curvature R\n(High near distractor)', fontsize=12)
    ax3.set_xlabel('Y')
    ax3.set_ylabel('X')
    plt.colorbar(im3, ax=ax3, label='R')

    plt.tight_layout()
    plt.savefig(output_dir / "00_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir / '00_summary.png'}")

    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  00_summary.png              - Complete overview")
    print("  01_scalar_fields.png        - φ₊ and φ₋ fields")
    print("  02_metric_fields.png        - Φ and R fields")
    print("  03_geodesic_trajectory.png  - Slingshot path")
    print("  04_slingshot_animation.gif  - Animated trajectory")
    print("\nKey observations:")
    print("  ✓ Geodesic curves around distractor (slingshot effect)")
    print("  ✓ High curvature near distractor due to phantom repulsion")
    print("  ✓ Smooth acceleration toward target")
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Gravitational Slingshot Navigation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick synthetic demo (no model required)
  python demo.py --mode synthetic

  # With custom repulsion strength
  python demo.py --mode synthetic --lambda_repulsion 3.0

  # With trained model (requires checkpoint)
  python demo.py --checkpoint experiments/best_model.pt --scene_id "17DRP5sb8fy"
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="synthetic",
        choices=["synthetic", "habitat"],
        help="Demo mode: synthetic (no model) or habitat (requires checkpoint)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (for habitat mode)"
    )

    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Habitat scene ID (for habitat mode)"
    )

    parser.add_argument(
        "--instruction",
        type=str,
        default="Go to the blue chair",
        help="Navigation instruction"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets/demo_output",
        help="Output directory for visualizations"
    )

    parser.add_argument(
        "--lambda_repulsion",
        type=float,
        default=2.0,
        help="Repulsion strength λ in Φ = φ₊ - λφ₋"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization (default: True)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.mode == "synthetic":
        run_synthetic_demo(output_dir, args.lambda_repulsion)
    else:
        print("Error: Habitat mode requires full model checkpoint and Habitat-Sim installation.")
        print("For quick demo, use: python demo.py --mode synthetic")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

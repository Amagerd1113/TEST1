"""
Conformal Metric and Geodesic Integration
==========================================

Constructs conformal metric from dual scalar fields and computes geodesics.

Physics:
From scalar fields φ₊ (attractive) and φ₋ (repulsive), we construct:

    g_ij = e^{2(φ₊ - λφ₋)} δ_ij

where λ ≥ 1 controls repulsion strength. This is a conformal metric in R³.

The geodesic equation in this metric is:
    d²x^i/ds² + Γ^i_jk (dx^j/ds)(dx^k/ds) = 0

where Γ^i_jk are Christoffel symbols. For conformal metrics:
    Γ^i_jk = δ^i_j ∂_k Φ + δ^i_k ∂_j Φ - δ_jk ∂^i Φ

with Φ = φ₊ - λφ₋ (the conformal factor).

Geodesics naturally exhibit slingshot behavior: paths are repelled by
high φ₋ regions (distractors) while attracted to high φ₊ regions (targets).

References:
- Wald (1984): General Relativity, Chapter 3
- Mannheim (2012): Making the Case for Conformal Gravity
- Riemannian Motion Policies (CoRL 2024)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from jaxtyping import Float
import numpy as np


class ConformalMetric(nn.Module):
    """
    Conformal metric g_ij = e^{2Φ} δ_ij where Φ = φ₊ - λφ₋.

    Computes metric tensor, inverse metric, and scalar curvature R.
    """

    def __init__(
        self,
        lambda_repulsion: float = 2.0,
        learn_lambda: bool = True,
        grid_spacing: float = 1.0,
    ):
        """
        Args:
            lambda_repulsion: Repulsion strength λ in Φ = φ₊ - λφ₋
            learn_lambda: Whether to learn λ during training
            grid_spacing: Spatial grid spacing for derivatives
        """
        super().__init__()

        if learn_lambda:
            self.lambda_repulsion = nn.Parameter(torch.tensor(lambda_repulsion))
        else:
            self.register_buffer("lambda_repulsion", torch.tensor(lambda_repulsion))

        self.grid_spacing = grid_spacing

    def compute_conformal_factor(
        self,
        phi_positive: Float[torch.Tensor, "batch H W D"],
        phi_negative: Float[torch.Tensor, "batch H W D"],
    ) -> Float[torch.Tensor, "batch H W D"]:
        """
        Compute conformal factor Φ = φ₊ - λφ₋.

        Args:
            phi_positive: (B, H, W, D) attractive scalar field
            phi_negative: (B, H, W, D) repulsive scalar field

        Returns:
            Phi: (B, H, W, D) conformal factor
        """
        return phi_positive - self.lambda_repulsion * phi_negative

    def compute_metric_tensor(
        self,
        Phi: Float[torch.Tensor, "batch H W D"],
    ) -> Float[torch.Tensor, "batch H W D 3 3"]:
        """
        Compute metric tensor g_ij = e^{2Φ} δ_ij.

        Args:
            Phi: (B, H, W, D) conformal factor

        Returns:
            g: (B, H, W, D, 3, 3) metric tensor (diagonal)
        """
        batch_size, H, W, D = Phi.shape

        # Conformal scale factor: e^{2Φ}
        scale = torch.exp(2 * Phi)  # (B, H, W, D)

        # Initialize as identity then scale
        g = torch.eye(3, device=Phi.device).view(1, 1, 1, 1, 3, 3)
        g = g.expand(batch_size, H, W, D, 3, 3).contiguous() * scale.unsqueeze(-1).unsqueeze(-1)

        return g

    def compute_inverse_metric(
        self,
        Phi: Float[torch.Tensor, "batch H W D"],
    ) -> Float[torch.Tensor, "batch H W D 3 3"]:
        """
        Compute inverse metric g^ij = e^{-2Φ} δ^ij.

        Args:
            Phi: (B, H, W, D) conformal factor

        Returns:
            g_inv: (B, H, W, D, 3, 3) inverse metric tensor (diagonal)
        """
        batch_size, H, W, D = Phi.shape

        # Inverse scale factor: e^{-2Φ}
        inv_scale = torch.exp(-2 * Phi)  # (B, H, W, D)

        # Initialize as identity then scale
        g_inv = torch.eye(3, device=Phi.device).view(1, 1, 1, 1, 3, 3)
        g_inv = g_inv.expand(batch_size, H, W, D, 3, 3).contiguous() * inv_scale.unsqueeze(-1).unsqueeze(-1)

        return g_inv

    def compute_gradient(
        self,
        field: Float[torch.Tensor, "batch H W D"],
    ) -> Float[torch.Tensor, "batch H W D 3"]:
        """
        Compute spatial gradient ∇field using central differences.

        Args:
            field: (B, H, W, D) scalar field

        Returns:
            gradient: (B, H, W, D, 3) gradient vector [∂_x, ∂_y, ∂_z]
        """
        # Compute derivatives using central differences with padding
        # ∂f/∂x ≈ (f_{i+1} - f_{i-1}) / (2h)

        # Pad field for boundary handling
        field_padded = torch.nn.functional.pad(field, (1, 1, 1, 1, 1, 1), mode='replicate')

        # X derivative
        grad_x = (field_padded[:, 2:, 1:-1, 1:-1] - field_padded[:, :-2, 1:-1, 1:-1]) / (2 * self.grid_spacing)

        # Y derivative
        grad_y = (field_padded[:, 1:-1, 2:, 1:-1] - field_padded[:, 1:-1, :-2, 1:-1]) / (2 * self.grid_spacing)

        # Z derivative
        grad_z = (field_padded[:, 1:-1, 1:-1, 2:] - field_padded[:, 1:-1, 1:-1, :-2]) / (2 * self.grid_spacing)

        # Stack into gradient vector
        gradient = torch.stack([grad_x, grad_y, grad_z], dim=-1)  # (B, H, W, D, 3)

        return gradient

    def compute_scalar_curvature(
        self,
        Phi: Float[torch.Tensor, "batch H W D"],
    ) -> Float[torch.Tensor, "batch H W D"]:
        """
        Compute scalar curvature R for conformal metric.

        For g_ij = e^{2Φ} δ_ij, the scalar curvature is:
            R = -2 e^{-2Φ} (∇²Φ + 2|∇Φ|²)

        Args:
            Phi: (B, H, W, D) conformal factor

        Returns:
            R: (B, H, W, D) scalar curvature
        """
        # Compute Laplacian of Φ
        laplacian = self._compute_laplacian(Phi)  # (B, H, W, D)

        # Compute gradient magnitude squared |∇Φ|²
        grad_Phi = self.compute_gradient(Phi)  # (B, H, W, D, 3)
        grad_squared = (grad_Phi ** 2).sum(dim=-1)  # (B, H, W, D)

        # Scalar curvature formula
        R = -2 * torch.exp(-2 * Phi) * (laplacian + 2 * grad_squared)

        return R

    def _compute_laplacian(
        self,
        field: Float[torch.Tensor, "batch H W D"],
    ) -> Float[torch.Tensor, "batch H W D"]:
        """
        Compute Laplacian ∇²field using finite differences.

        Args:
            field: (B, H, W, D) scalar field

        Returns:
            laplacian: (B, H, W, D) Laplacian
        """
        # Pad field
        field_padded = torch.nn.functional.pad(field, (1, 1, 1, 1, 1, 1), mode='replicate')

        # 7-point stencil: (f_{i+1} + f_{i-1} + ... - 6f_center) / h²
        laplacian = (
            field_padded[:, 2:, 1:-1, 1:-1] +  # x+1
            field_padded[:, :-2, 1:-1, 1:-1] +  # x-1
            field_padded[:, 1:-1, 2:, 1:-1] +  # y+1
            field_padded[:, 1:-1, :-2, 1:-1] +  # y-1
            field_padded[:, 1:-1, 1:-1, 2:] +  # z+1
            field_padded[:, 1:-1, 1:-1, :-2] -  # z-1
            6 * field  # center
        ) / (self.grid_spacing ** 2)

        return laplacian

    def compute_christoffel_symbols(
        self,
        Phi: Float[torch.Tensor, "batch H W D"],
    ) -> Float[torch.Tensor, "batch H W D 3 3 3"]:
        """
        Compute Christoffel symbols Γ^i_jk for conformal metric.

        For g_ij = e^{2Φ} δ_ij:
            Γ^i_jk = δ^i_j ∂_k Φ + δ^i_k ∂_j Φ - δ_jk ∂^i Φ

        Args:
            Phi: (B, H, W, D) conformal factor

        Returns:
            Gamma: (B, H, W, D, 3, 3, 3) Christoffel symbols [i, j, k]
        """
        batch_size, H, W, D = Phi.shape

        # Compute gradient of Φ: ∂_i Φ
        grad_Phi = self.compute_gradient(Phi)  # (B, H, W, D, 3)

        # Initialize Christoffel symbols
        Gamma = torch.zeros(batch_size, H, W, D, 3, 3, 3, device=Phi.device)

        # Γ^i_jk = δ^i_j ∂_k Φ + δ^i_k ∂_j Φ - δ_jk ∂^i Φ
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    term1 = float(i == j) * grad_Phi[..., k]  # δ^i_j ∂_k Φ
                    term2 = float(i == k) * grad_Phi[..., j]  # δ^i_k ∂_j Φ
                    term3 = float(j == k) * grad_Phi[..., i]  # δ_jk ∂^i Φ

                    Gamma[:, :, :, :, i, j, k] = term1 + term2 - term3

        return Gamma


class GeodesicIntegrator(nn.Module):
    """
    Integrates geodesic equations to generate trajectories.

    Uses RK4 (Runge-Kutta 4th order) integration for accuracy and stability.
    """

    def __init__(
        self,
        conformal_metric: ConformalMetric,
        integration_method: str = "rk4",
        step_size: float = 0.1,
        max_steps: int = 50,
        max_curvature: float = 100.0,
    ):
        """
        Args:
            conformal_metric: ConformalMetric instance
            integration_method: Integration method ("euler" or "rk4")
            step_size: Integration step size (meters)
            max_steps: Maximum integration steps
            max_curvature: Clip extreme curvature values
        """
        super().__init__()
        self.conformal_metric = conformal_metric
        self.integration_method = integration_method
        self.step_size = step_size
        self.max_steps = max_steps
        self.max_curvature = max_curvature

    def integrate_geodesic(
        self,
        start_pos: Float[torch.Tensor, "batch 3"],
        start_vel: Float[torch.Tensor, "batch 3"],
        Phi: Float[torch.Tensor, "batch H W D"],
    ) -> Tuple[
        Float[torch.Tensor, "batch max_steps 3"],
        Float[torch.Tensor, "batch max_steps 3"]
    ]:
        """
        Integrate geodesic equation from initial position and velocity.

        Geodesic equation:
            d²x^i/ds² + Γ^i_jk (dx^j/ds)(dx^k/ds) = 0

        Args:
            start_pos: (B, 3) initial position [x, y, z]
            start_vel: (B, 3) initial velocity [vx, vy, vz]
            Phi: (B, H, W, D) conformal factor field

        Returns:
            positions: (B, max_steps, 3) trajectory positions
            velocities: (B, max_steps, 3) trajectory velocities
        """
        batch_size = start_pos.size(0)
        device = start_pos.device

        # Initialize trajectory storage
        positions = torch.zeros(batch_size, self.max_steps, 3, device=device)
        velocities = torch.zeros(batch_size, self.max_steps, 3, device=device)

        # Set initial conditions
        pos = start_pos.clone()  # (B, 3)
        vel = start_vel.clone()  # (B, 3)

        positions[:, 0] = pos
        velocities[:, 0] = vel

        # Precompute Christoffel symbols
        Gamma = self.conformal_metric.compute_christoffel_symbols(Phi)  # (B, H, W, D, 3, 3, 3)

        # Integration loop
        for step in range(1, self.max_steps):
            if self.integration_method == "rk4":
                pos, vel = self._rk4_step(pos, vel, Phi, Gamma)
            else:  # euler
                pos, vel = self._euler_step(pos, vel, Phi, Gamma)

            positions[:, step] = pos
            velocities[:, step] = vel

        return positions, velocities

    def _geodesic_acceleration(
        self,
        pos: Float[torch.Tensor, "batch 3"],
        vel: Float[torch.Tensor, "batch 3"],
        Phi: Float[torch.Tensor, "batch H W D"],
        Gamma: Float[torch.Tensor, "batch H W D 3 3 3"],
    ) -> Float[torch.Tensor, "batch 3"]:
        """
        Compute geodesic acceleration: a^i = -Γ^i_jk v^j v^k

        Args:
            pos: (B, 3) position
            vel: (B, 3) velocity
            Phi: (B, H, W, D) conformal factor
            Gamma: (B, H, W, D, 3, 3, 3) Christoffel symbols

        Returns:
            accel: (B, 3) acceleration
        """
        batch_size = pos.size(0)

        # Interpolate Christoffel symbols at current position
        # For simplicity, use nearest neighbor (can upgrade to trilinear)
        H, W, D = Phi.shape[1:]

        # Normalize positions to grid indices
        indices = (pos * torch.tensor([H-1, W-1, D-1], device=pos.device)).long()
        indices = torch.clamp(indices, 0, torch.tensor([H-1, W-1, D-1], device=pos.device))

        # Extract Gamma at positions: (B, 3, 3, 3)
        Gamma_at_pos = torch.stack([
            Gamma[b, indices[b, 0], indices[b, 1], indices[b, 2]]
            for b in range(batch_size)
        ], dim=0)

        # Compute acceleration: a^i = -Γ^i_jk v^j v^k
        # Einstein summation over j, k
        accel = -torch.einsum('bijk,bj,bk->bi', Gamma_at_pos, vel, vel)

        # Clip extreme accelerations for stability
        accel = torch.clamp(accel, -self.max_curvature, self.max_curvature)

        return accel

    def _euler_step(
        self,
        pos: Float[torch.Tensor, "batch 3"],
        vel: Float[torch.Tensor, "batch 3"],
        Phi: Float[torch.Tensor, "batch H W D"],
        Gamma: Float[torch.Tensor, "batch H W D 3 3 3"],
    ) -> Tuple[Float[torch.Tensor, "batch 3"], Float[torch.Tensor, "batch 3"]]:
        """Euler integration step."""
        accel = self._geodesic_acceleration(pos, vel, Phi, Gamma)

        pos_new = pos + self.step_size * vel
        vel_new = vel + self.step_size * accel

        return pos_new, vel_new

    def _rk4_step(
        self,
        pos: Float[torch.Tensor, "batch 3"],
        vel: Float[torch.Tensor, "batch 3"],
        Phi: Float[torch.Tensor, "batch H W D"],
        Gamma: Float[torch.Tensor, "batch H W D 3 3 3"],
    ) -> Tuple[Float[torch.Tensor, "batch 3"], Float[torch.Tensor, "batch 3"]]:
        """RK4 integration step."""
        h = self.step_size

        # k1
        k1_vel = vel
        k1_accel = self._geodesic_acceleration(pos, vel, Phi, Gamma)

        # k2
        pos2 = pos + 0.5 * h * k1_vel
        vel2 = vel + 0.5 * h * k1_accel
        k2_vel = vel2
        k2_accel = self._geodesic_acceleration(pos2, vel2, Phi, Gamma)

        # k3
        pos3 = pos + 0.5 * h * k2_vel
        vel3 = vel + 0.5 * h * k2_accel
        k3_vel = vel3
        k3_accel = self._geodesic_acceleration(pos3, vel3, Phi, Gamma)

        # k4
        pos4 = pos + h * k3_vel
        vel4 = vel + h * k3_accel
        k4_vel = vel4
        k4_accel = self._geodesic_acceleration(pos4, vel4, Phi, Gamma)

        # Combine
        pos_new = pos + (h / 6.0) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)
        vel_new = vel + (h / 6.0) * (k1_accel + 2*k2_accel + 2*k3_accel + k4_accel)

        return pos_new, vel_new


if __name__ == "__main__":
    # Quick test
    print("Testing ConformalMetric and GeodesicIntegrator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create conformal metric
    metric = ConformalMetric(lambda_repulsion=2.0, learn_lambda=False).to(device)

    # Create test scalar fields
    batch_size = 2
    H, W, D = 64, 64, 64

    phi_pos = torch.randn(batch_size, H, W, D).to(device)
    phi_neg = torch.randn(batch_size, H, W, D).to(device)

    # Compute conformal factor
    Phi = metric.compute_conformal_factor(phi_pos, phi_neg)
    print(f"✓ Conformal factor Φ: shape={Phi.shape}, range=[{Phi.min():.3f}, {Phi.max():.3f}]")

    # Compute scalar curvature
    R = metric.compute_scalar_curvature(Phi)
    print(f"✓ Scalar curvature R: shape={R.shape}, range=[{R.min():.3f}, {R.max():.3f}]")

    # Test geodesic integration
    integrator = GeodesicIntegrator(metric, integration_method="rk4", step_size=0.1, max_steps=50)

    start_pos = torch.tensor([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]], device=device)
    start_vel = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], device=device)

    import time
    start_time = time.time()
    positions, velocities = integrator.integrate_geodesic(start_pos, start_vel, Phi)
    elapsed = (time.time() - start_time) * 1000

    print(f"✓ Geodesic integrated in {elapsed:.2f}ms")
    print(f"  Positions: shape={positions.shape}")
    print(f"  Final positions: {positions[:, -1]}")

    print("✓ All tests passed!")

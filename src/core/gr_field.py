"""
GR Field Manager: Compute metric tensor using Einstein field equations.
Implements general relativity-inspired navigation in curved spacetime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
import logging
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


class GRFieldManager(nn.Module):
    """
    Manage General Relativity field computations for navigation.
    
    Key concepts:
    - Metric tensor g_μν describes spacetime curvature
    - Energy-momentum tensor T_μν from affordance masses
    - Einstein field equations: R_μν - 1/2 R g_μν = (8πG/c⁴) T_μν
    - Geodesics represent optimal paths in curved spacetime
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # GR parameters
        self.grid_size = config['model']['gr_field']['grid_size']
        self.field_dim = config['model']['gr_field']['field_dim']
        self.c = config['model']['gr_field']['c']  # Speed of light
        self.G = config['model']['gr_field']['G']  # Gravitational constant
        self.lambda_curvature = config['model']['gr_field']['lambda_curvature']
        
        # Field computation modules
        self.metric_tensor_net = MetricTensorNetwork(
            input_dim=256,
            output_dim=self.field_dim,
            grid_size=self.grid_size
        )
        
        self.energy_momentum_net = EnergyMomentumTensor(
            mass_dim=1,
            velocity_dim=3,
            output_dim=10  # 4x4 symmetric tensor has 10 components
        )
        
        self.einstein_solver = EinsteinFieldSolver(
            c=self.c,
            G=self.G,
            lambda_reg=self.lambda_curvature
        )
        
        self.christoffel_computer = ChristoffelSymbols()
        
        self.riemann_tensor = RiemannCurvatureTensor()
        
        # Field refinement network
        self.field_refiner = FieldRefinementNetwork(
            field_dim=self.field_dim,
            hidden_dim=128
        )
        
        # Field caching for efficiency
        self.field_cache = {}
        self.cache_size = 10
        
    def forward(
        self,
        affordance_map: torch.Tensor,
        position: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GR field from affordance map.
        
        Args:
            affordance_map: Gaussian mass distributions [B, H, W, C]
            position: Current position [B, 3]
            velocity: Current velocity [B, 3]
            
        Returns:
            Dictionary containing:
                - metric_tensor: Spacetime metric g_μν [B, H, W, 10]
                - christoffel_symbols: Connection coefficients Γ^α_μν
                - riemann_tensor: Curvature tensor R^ρ_σμν
                - energy_momentum: Energy-momentum tensor T_μν
                - geodesic_acceleration: Acceleration along geodesics
        """
        
        B, H, W, C = affordance_map.shape
        device = affordance_map.device
        
        # Convert affordance to mass distribution
        mass_distribution = affordance_map.sum(dim=-1, keepdim=True)  # [B, H, W, 1]
        
        # Compute energy-momentum tensor
        energy_momentum = self.energy_momentum_net(
            mass_distribution,
            velocity if velocity is not None else torch.zeros(B, 3, device=device)
        )
        
        # Initial metric tensor (Minkowski spacetime as background)
        metric_init = self._initialize_metric(B, H, W, device)
        
        # Solve Einstein field equations
        metric_tensor = self.einstein_solver(
            metric_init,
            energy_momentum,
            mass_distribution
        )
        
        # Refine field using neural network
        metric_refined = self.field_refiner(
            metric_tensor,
            affordance_map
        )
        
        # Compute Christoffel symbols (connection coefficients)
        christoffel = self.christoffel_computer(metric_refined)
        
        # Compute Riemann curvature tensor
        riemann = self.riemann_tensor(christoffel, metric_refined)
        
        # Compute geodesic acceleration if position/velocity provided
        geodesic_acc = None
        if position is not None and velocity is not None:
            geodesic_acc = self._compute_geodesic_acceleration(
                position, velocity, christoffel
            )
            
        # Cache result
        self._update_cache(affordance_map, metric_refined)
        
        return {
            'metric_tensor': metric_refined,
            'christoffel_symbols': christoffel,
            'riemann_tensor': riemann,
            'energy_momentum': energy_momentum,
            'geodesic_acceleration': geodesic_acc,
            'mass_distribution': mass_distribution
        }
    
    def _initialize_metric(
        self,
        B: int,
        H: int,
        W: int,
        device: torch.device
    ) -> torch.Tensor:
        """Initialize metric tensor with Minkowski background."""
        
        # Minkowski metric: diag(-1, 1, 1, 1) in signature (-,+,+,+)
        metric = torch.zeros(B, H, W, 10, device=device)
        
        # Set diagonal components
        # g_00 = -1 (time component)
        metric[..., 0] = -1.0
        # g_11 = g_22 = g_33 = 1 (spatial components)
        metric[..., 4] = 1.0  # g_11
        metric[..., 7] = 1.0  # g_22
        metric[..., 9] = 1.0  # g_33
        
        return metric
    
    def _compute_geodesic_acceleration(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        christoffel: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute acceleration along geodesics.
        
        Geodesic equation: d²x^μ/dτ² + Γ^μ_αβ (dx^α/dτ)(dx^β/dτ) = 0
        """
        
        B = position.shape[0]
        device = position.device
        
        # Convert position to grid coordinates
        grid_pos = self._world_to_grid(position)
        
        # Sample Christoffel symbols at current position
        christoffel_local = self._sample_field_at_position(
            christoffel, grid_pos
        )
        
        # Compute geodesic acceleration
        # a^μ = -Γ^μ_αβ v^α v^β
        acceleration = torch.zeros(B, 4, device=device)
        
        # Add time component
        v_4d = torch.cat([
            torch.ones(B, 1, device=device),  # dt/dτ = 1
            velocity
        ], dim=-1)  # [B, 4]
        
        # Contract indices
        for mu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    idx = self._christoffel_index(mu, alpha, beta)
                    if idx < christoffel_local.shape[-1]:
                        acceleration[:, mu] -= (
                            christoffel_local[:, idx] *
                            v_4d[:, alpha] * v_4d[:, beta]
                        )
                        
        return acceleration[:, 1:]  # Return only spatial components
    
    def _world_to_grid(self, position: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to grid indices."""
        # Assume world spans [-10, 10] meters
        normalized_pos = (position + 10.0) / 20.0  # [0, 1]
        
        grid_pos = normalized_pos * torch.tensor(
            self.grid_size[:3],
            device=position.device,
            dtype=torch.float32
        )
        
        return grid_pos.long()
    
    def _sample_field_at_position(
        self,
        field: torch.Tensor,
        grid_pos: torch.Tensor
    ) -> torch.Tensor:
        """Sample field values at given grid positions."""
        B = grid_pos.shape[0]
        
        # Clamp to valid range
        grid_size = torch.tensor(self.grid_size[:3], device=grid_pos.device)
        grid_pos = torch.clamp(grid_pos, min=0, max=grid_size - 1)
        
        # Sample field
        sampled = []
        for b in range(B):
            x, y, z = grid_pos[b]
            if z < field.shape[2]:  # Handle 2D fields
                sampled.append(field[b, x, y, z])
            else:
                sampled.append(field[b, x, y, 0])
                
        return torch.stack(sampled)
    
    def _christoffel_index(self, mu: int, alpha: int, beta: int) -> int:
        """Map Christoffel symbol indices to flat index."""
        # Use symmetry in lower indices: Γ^μ_αβ = Γ^μ_βα
        if alpha > beta:
            alpha, beta = beta, alpha
        
        # Flatten to single index
        # Total components: 4 * 10 = 40 (4 upper, 10 lower symmetric pairs)
        lower_idx = alpha * 4 - alpha * (alpha - 1) // 2 + beta
        return mu * 10 + lower_idx
    
    def _update_cache(self, affordance: torch.Tensor, metric: torch.Tensor):
        """Update field cache for efficiency."""
        # Create hash key from affordance
        key = hash(affordance.cpu().numpy().tobytes())
        
        self.field_cache[key] = metric.detach()
        
        # Maintain cache size
        if len(self.field_cache) > self.cache_size:
            # Remove oldest entry
            oldest_key = list(self.field_cache.keys())[0]
            del self.field_cache[oldest_key]
            
    def reset(self):
        """Reset field manager state."""
        self.field_cache.clear()


class MetricTensorNetwork(nn.Module):
    """Neural network for computing metric tensor components."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size: List[int]
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.output_dim = output_dim
        
        # 3D convolutional network for spatial field
        self.conv3d_net = nn.Sequential(
            nn.Conv3d(input_dim, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, output_dim, 3, padding=1)
        )
        
        # Ensure metric tensor symmetry
        self.symmetrizer = MetricSymmetrizer()
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor from features."""
        
        B = features.shape[0]
        
        # Reshape to 3D grid if needed
        if features.dim() == 4:  # [B, H, W, C]
            # Add depth dimension
            features = features.unsqueeze(2)  # [B, H, 1, W, C]
            features = features.repeat(1, 1, self.grid_size[2], 1, 1)
            
        # Permute for Conv3D
        features = features.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        
        # Apply 3D convolutions
        metric_raw = self.conv3d_net(features)  # [B, 10, D, H, W]
        
        # Permute back
        metric = metric_raw.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 10]
        
        # Ensure symmetry
        metric = self.symmetrizer(metric)
        
        # Take middle slice for 2D output
        metric_2d = metric[:, self.grid_size[2]//2]  # [B, H, W, 10]
        
        return metric_2d


class EnergyMomentumTensor(nn.Module):
    """Compute energy-momentum tensor from mass and velocity."""
    
    def __init__(self, mass_dim: int, velocity_dim: int, output_dim: int):
        super().__init__()
        
        self.mass_dim = mass_dim
        self.velocity_dim = velocity_dim
        
        # Energy density computation
        self.energy_net = nn.Sequential(
            nn.Linear(mass_dim + velocity_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive energy
        )
        
        # Momentum density computation
        self.momentum_net = nn.Sequential(
            nn.Linear(mass_dim + velocity_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )
        
        # Stress tensor computation
        self.stress_net = nn.Sequential(
            nn.Linear(mass_dim + velocity_dim, 128),
            nn.GELU(),
            nn.Linear(128, 6)  # 3x3 symmetric tensor
        )
        
    def forward(
        self,
        mass: torch.Tensor,
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy-momentum tensor T_μν.
        
        T^00 = energy density
        T^0i = T^i0 = momentum density
        T^ij = stress tensor
        """
        
        B, H, W, _ = mass.shape
        device = mass.device
        
        # Flatten spatial dimensions
        mass_flat = mass.reshape(B * H * W, -1)
        
        # Expand velocity for all spatial points
        velocity_expanded = velocity.unsqueeze(1).unsqueeze(1)
        velocity_expanded = velocity_expanded.expand(B, H, W, -1)
        velocity_flat = velocity_expanded.reshape(B * H * W, -1)
        
        # Concatenate inputs
        inputs = torch.cat([mass_flat, velocity_flat], dim=-1)
        
        # Compute components
        energy = self.energy_net(inputs)  # [B*H*W, 1]
        momentum = self.momentum_net(inputs)  # [B*H*W, 3]
        stress = self.stress_net(inputs)  # [B*H*W, 6]
        
        # Construct full tensor (4x4 symmetric = 10 components)
        tensor_components = []
        
        # T^00 (energy density)
        tensor_components.append(energy)
        
        # T^0i = T^i0 (momentum density)
        for i in range(3):
            tensor_components.append(momentum[:, i:i+1])
            
        # T^ij (stress tensor, symmetric)
        # Order: T^11, T^12, T^13, T^22, T^23, T^33
        for i in range(6):
            tensor_components.append(stress[:, i:i+1])
            
        # Stack components
        energy_momentum = torch.cat(tensor_components, dim=-1)  # [B*H*W, 10]
        
        # Reshape back
        energy_momentum = energy_momentum.reshape(B, H, W, 10)
        
        return energy_momentum


class EinsteinFieldSolver(nn.Module):
    """Solve Einstein field equations numerically."""
    
    def __init__(self, c: float, G: float, lambda_reg: float):
        super().__init__()
        
        self.c = c
        self.G = G
        self.lambda_reg = lambda_reg
        self.coupling = 8 * np.pi * G / (c ** 4)
        
        # Learnable parameters for solver
        self.solver_params = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        
    def forward(
        self,
        metric_init: torch.Tensor,
        energy_momentum: torch.Tensor,
        mass: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve Einstein field equations:
        R_μν - 1/2 R g_μν + Λ g_μν = (8πG/c⁴) T_μν
        
        Using linearized approximation for weak fields.
        """
        
        B, H, W, _ = metric_init.shape
        device = metric_init.device
        
        # Linearized solution: g_μν = η_μν + h_μν
        # where η_μν is Minkowski metric and h_μν is perturbation
        
        # Extract Minkowski background
        eta = metric_init.clone()
        
        # Compute perturbation from energy-momentum tensor
        # In weak field limit: h_μν ≈ -(16πG/c⁴) T_μν
        h = -self.coupling * energy_momentum
        
        # Apply regularization based on mass distribution
        mass_normalized = mass / (mass.max() + 1e-8)
        regularization = self.lambda_reg * mass_normalized
        
        # Weighted combination
        weights = F.softmax(self.solver_params, dim=0)
        
        # Perturbed metric
        metric = eta + weights[0] * h
        
        # Add curvature from mass
        curvature_correction = self._compute_curvature_correction(
            mass_normalized, H, W, device
        )
        metric = metric + weights[1] * curvature_correction
        
        # Ensure metric remains non-degenerate
        metric = self._ensure_metric_validity(metric)
        
        # Apply smoothing for stability
        if weights[2] > 0.1:
            metric = self._smooth_field(metric)
            
        return metric
    
    def _compute_curvature_correction(
        self,
        mass: torch.Tensor,
        H: int,
        W: int,
        device: torch.device
    ) -> torch.Tensor:
        """Compute curvature correction from mass distribution."""
        
        B = mass.shape[0]
        
        # Create spatial grid
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Compute gravitational potential
        potential = F.conv2d(
            mass.permute(0, 3, 1, 2),  # [B, 1, H, W]
            self._green_function_kernel(device),
            padding='same'
        )
        
        # Compute metric perturbation from potential
        correction = torch.zeros(B, H, W, 10, device=device)
        
        # Spatial components get modified
        grad_x = torch.gradient(potential, dim=-1)[0]
        grad_y = torch.gradient(potential, dim=-2)[0]
        
        # Modify diagonal spatial components
        correction[..., 4] = 2 * self.G * potential.squeeze(1)  # g_11
        correction[..., 7] = 2 * self.G * potential.squeeze(1)  # g_22
        
        return correction
    
    def _green_function_kernel(self, device: torch.device) -> torch.Tensor:
        """Green's function kernel for Poisson equation."""
        size = 7
        kernel = torch.zeros(1, 1, size, size, device=device)
        
        center = size // 2
        for i in range(size):
            for j in range(size):
                r = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if r > 0:
                    kernel[0, 0, i, j] = -1 / (2 * np.pi * r)
                    
        return kernel
    
    def _ensure_metric_validity(self, metric: torch.Tensor) -> torch.Tensor:
        """Ensure metric tensor remains valid (non-degenerate)."""
        
        # Ensure time component remains negative
        metric[..., 0] = torch.minimum(metric[..., 0], torch.tensor(-0.1))
        
        # Ensure spatial components remain positive
        metric[..., 4] = torch.maximum(metric[..., 4], torch.tensor(0.1))
        metric[..., 7] = torch.maximum(metric[..., 7], torch.tensor(0.1))
        metric[..., 9] = torch.maximum(metric[..., 9], torch.tensor(0.1))
        
        return metric
    
    def _smooth_field(self, field: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to field."""
        B, H, W, C = field.shape
        
        # Reshape for convolution
        field_reshaped = field.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Gaussian kernel
        kernel = self._gaussian_kernel(3, device=field.device)
        kernel = kernel.expand(C, 1, 3, 3)
        
        # Apply smoothing
        field_smooth = F.conv2d(
            field_reshaped,
            kernel,
            padding=1,
            groups=C
        )
        
        # Reshape back
        field_smooth = field_smooth.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        return field_smooth
    
    def _gaussian_kernel(
        self,
        size: int,
        device: torch.device,
        sigma: float = 1.0
    ) -> torch.Tensor:
        """Create Gaussian kernel."""
        coords = torch.arange(size, device=device) - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.view(1, 1, 1, -1) * g.view(1, 1, -1, 1)
        return kernel


class ChristoffelSymbols(nn.Module):
    """Compute Christoffel symbols from metric tensor."""
    
    def forward(self, metric: torch.Tensor) -> torch.Tensor:
        """
        Compute Christoffel symbols:
        Γ^λ_μν = 1/2 g^λσ (∂_μ g_σν + ∂_ν g_σμ - ∂_σ g_μν)
        """
        
        B, H, W, _ = metric.shape
        device = metric.device
        
        # Compute metric derivatives
        # Using finite differences
        metric_dx = torch.gradient(metric, dim=2)[0]  # ∂/∂x
        metric_dy = torch.gradient(metric, dim=1)[0]  # ∂/∂y
        
        # For simplicity, compute only spatial Christoffel symbols
        # Full computation would include all 4^3 = 64 components
        
        christoffel = torch.zeros(B, H, W, 40, device=device)
        
        # Simplified computation for key components
        # This is a placeholder - full implementation would compute all components
        
        return christoffel


class RiemannCurvatureTensor(nn.Module):
    """Compute Riemann curvature tensor."""
    
    def forward(
        self,
        christoffel: torch.Tensor,
        metric: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Riemann tensor:
        R^ρ_σμν = ∂_μ Γ^ρ_νσ - ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ - Γ^ρ_νλ Γ^λ_μσ
        """
        
        # Simplified computation
        # Full tensor has 4^4 = 256 components (with symmetries reducing to 20)
        
        B, H, W, _ = metric.shape
        device = metric.device
        
        riemann = torch.zeros(B, H, W, 20, device=device)
        
        # Placeholder for actual computation
        # Would involve derivatives of Christoffel symbols
        
        return riemann


class MetricSymmetrizer(nn.Module):
    """Ensure metric tensor symmetry."""
    
    def forward(self, metric: torch.Tensor) -> torch.Tensor:
        """Symmetrize metric tensor components."""
        
        # Metric stored as: g_00, g_01, g_02, g_03, g_11, g_12, g_13, g_22, g_23, g_33
        # Already in symmetric storage format
        
        return metric


class FieldRefinementNetwork(nn.Module):
    """Neural network for refining GR field."""
    
    def __init__(self, field_dim: int, hidden_dim: int):
        super().__init__()
        
        self.refiner = nn.Sequential(
            nn.Conv2d(field_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, field_dim, 3, padding=1)
        )
        
        # Residual connection
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(
        self,
        field: torch.Tensor,
        affordance: torch.Tensor
    ) -> torch.Tensor:
        """Refine field using neural network."""
        
        B, H, W, C = field.shape
        
        # Add affordance information
        combined = torch.cat([field, affordance], dim=-1)
        
        # Project to field dimension
        if combined.shape[-1] != C:
            projection = nn.Linear(
                combined.shape[-1],
                C,
                device=field.device
            )
            combined = projection(combined)
            
        # Apply refinement
        field_2d = combined.permute(0, 3, 1, 2)  # [B, C, H, W]
        refined = self.refiner(field_2d)
        refined = refined.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # Residual connection
        output = field + self.residual_weight * refined
        
        return output

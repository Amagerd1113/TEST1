"""
Path Optimizer: Compute optimal trajectories as geodesics in curved spacetime.
Implements physics-constrained path planning using variational principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class PathOptimizer(nn.Module):
    """
    Optimize robot trajectories as geodesics in GR field.
    
    Key concepts:
    - Geodesics minimize proper time in curved spacetime
    - Variational principle: δ∫ ds = 0
    - Physical constraints: velocity, acceleration limits
    - Dynamic replanning with receding horizon
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Path parameters
        self.horizon = config['model']['path']['horizon']
        self.dt = config['model']['path']['dt']
        self.max_velocity = config['model']['path']['max_velocity']
        self.max_acceleration = config['model']['path']['max_acceleration']
        self.collision_radius = config['model']['path']['collision_radius']
        self.geodesic_samples = config['model']['path']['geodesic_samples']
        self.optimization_steps = config['model']['path']['optimization_steps']
        
        # Geodesic solver
        self.geodesic_solver = GeodesicSolver(
            num_samples=self.geodesic_samples,
            max_iterations=self.optimization_steps
        )
        
        # Goal encoder
        self.goal_encoder = GoalEncoder(
            language_dim=config['model']['language']['embed_dim'],
            output_dim=256
        )
        
        # Trajectory refinement network
        self.trajectory_refiner = TrajectoryRefinementNetwork(
            hidden_dim=256,
            num_waypoints=self.horizon
        )
        
        # Constraint enforcer
        self.constraint_enforcer = PhysicsConstraints(
            max_vel=self.max_velocity,
            max_acc=self.max_acceleration,
            dt=self.dt
        )
        
        # Collision checker
        self.collision_checker = CollisionChecker(
            radius=self.collision_radius
        )
        
        # Path scoring network
        self.path_scorer = PathScoringNetwork(
            input_dim=256,
            hidden_dim=128
        )
        
    def forward(
        self,
        gr_field: torch.Tensor,
        start_position: torch.Tensor,
        goal_features: torch.Tensor,
        affordance_map: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Optimize path through curved spacetime.
        
        Args:
            gr_field: Metric tensor field [B, H, W, 10]
            start_position: Starting position [B, 3]
            goal_features: Encoded goal from language [B, L, D]
            affordance_map: Affordance distributions [B, H, W, C]
            
        Returns:
            Dictionary containing:
                - trajectory: Optimized path [B, T, 3]
                - waypoints: Key points along path [B, N, 3]
                - cost: Path cost/length
                - velocity_profile: Velocity along path [B, T, 3]
                - valid: Whether path is collision-free
        """
        
        B = gr_field.shape[0]
        device = gr_field.device
        
        # Encode goal position from language features
        goal_position = self.goal_encoder(goal_features)  # [B, 3]
        
        # Sample initial path candidates
        path_candidates = self._sample_initial_paths(
            start_position, goal_position, B, device
        )
        
        # Optimize each candidate as geodesic
        optimized_paths = []
        path_costs = []
        
        for candidate in path_candidates:
            # Solve geodesic equation
            geodesic = self.geodesic_solver(
                path=candidate,
                metric_field=gr_field,
                affordance=affordance_map
            )
            
            # Apply physics constraints
            constrained_path = self.constraint_enforcer(geodesic)
            
            # Check collisions
            valid = self.collision_checker(constrained_path, affordance_map)
            
            if valid:
                optimized_paths.append(constrained_path)
                
                # Compute path cost
                cost = self._compute_path_cost(
                    constrained_path,
                    gr_field,
                    affordance_map
                )
                path_costs.append(cost)
                
        # Select best path
        if optimized_paths:
            path_costs_tensor = torch.stack(path_costs)
            best_idx = torch.argmin(path_costs_tensor, dim=0)
            
            best_paths = []
            for b in range(B):
                best_paths.append(optimized_paths[best_idx[b]][b:b+1])
            best_path = torch.cat(best_paths, dim=0)
        else:
            # Fallback to straight line if no valid path found
            logger.warning("No valid geodesic found, using straight line")
            best_path = self._straight_line_path(
                start_position, goal_position, self.horizon
            )
            
        # Refine trajectory using neural network
        refined_trajectory = self.trajectory_refiner(
            trajectory=best_path,
            gr_field=gr_field,
            affordance=affordance_map,
            goal_features=goal_features
        )
        
        # Compute velocity profile
        velocity_profile = self._compute_velocity_profile(refined_trajectory)
        
        # Extract waypoints
        waypoints = self._extract_waypoints(refined_trajectory, num_waypoints=10)
        
        # Final validation
        is_valid = self.collision_checker(refined_trajectory, affordance_map)
        
        # Compute final cost
        final_cost = self._compute_path_cost(
            refined_trajectory,
            gr_field,
            affordance_map
        )
        
        return {
            'trajectory': refined_trajectory,
            'waypoints': waypoints,
            'cost': final_cost,
            'velocity_profile': velocity_profile,
            'valid': is_valid,
            'goal_position': goal_position
        }
    
    def _sample_initial_paths(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        batch_size: int,
        device: torch.device
    ) -> List[torch.Tensor]:
        """Sample diverse initial path candidates."""
        
        paths = []
        
        # Straight line path
        straight_path = self._straight_line_path(start, goal, self.horizon)
        paths.append(straight_path)
        
        # Curved paths with different curvatures
        for curve_factor in [0.1, 0.2, 0.3, -0.1, -0.2]:
            curved_path = self._curved_path(
                start, goal, self.horizon, curve_factor
            )
            paths.append(curved_path)
            
        # Random perturbations
        for _ in range(3):
            random_path = self._random_path(start, goal, self.horizon, device)
            paths.append(random_path)
            
        return paths
    
    def _straight_line_path(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        num_points: int
    ) -> torch.Tensor:
        """Generate straight line path."""
        
        B = start.shape[0]
        device = start.device
        
        # Interpolate linearly
        t = torch.linspace(0, 1, num_points, device=device)
        t = t.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        
        start_expanded = start.unsqueeze(1)  # [B, 1, 3]
        goal_expanded = goal.unsqueeze(1)  # [B, 1, 3]
        
        path = start_expanded * (1 - t) + goal_expanded * t  # [B, T, 3]
        
        return path
    
    def _curved_path(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        num_points: int,
        curve_factor: float
    ) -> torch.Tensor:
        """Generate curved path with specified curvature."""
        
        B = start.shape[0]
        device = start.device
        
        # Add control point for curve
        midpoint = (start + goal) / 2
        
        # Perpendicular direction
        direction = goal - start
        perpendicular = torch.stack([
            -direction[:, 1],
            direction[:, 0],
            torch.zeros_like(direction[:, 0])
        ], dim=-1)
        
        # Control point
        control = midpoint + curve_factor * perpendicular
        
        # Quadratic Bezier curve
        t = torch.linspace(0, 1, num_points, device=device)
        t = t.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        
        start_expanded = start.unsqueeze(1)  # [B, 1, 3]
        control_expanded = control.unsqueeze(1)  # [B, 1, 3]
        goal_expanded = goal.unsqueeze(1)  # [B, 1, 3]
        
        path = ((1 - t) ** 2) * start_expanded + \
               2 * (1 - t) * t * control_expanded + \
               (t ** 2) * goal_expanded
        
        return path
    
    def _random_path(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        num_points: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate random path with smoothing."""
        
        B = start.shape[0]
        
        # Random waypoints
        num_waypoints = 5
        waypoints = []
        waypoints.append(start)
        
        for i in range(1, num_waypoints - 1):
            alpha = i / (num_waypoints - 1)
            base_point = start * (1 - alpha) + goal * alpha
            noise = torch.randn(B, 3, device=device) * 0.5
            waypoints.append(base_point + noise)
            
        waypoints.append(goal)
        
        # Interpolate with cubic spline
        # For simplicity, using linear interpolation here
        waypoints_tensor = torch.stack(waypoints, dim=1)  # [B, W, 3]
        
        # Upsample to full trajectory
        path = F.interpolate(
            waypoints_tensor.transpose(1, 2),  # [B, 3, W]
            size=num_points,
            mode='linear',
            align_corners=True
        ).transpose(1, 2)  # [B, T, 3]
        
        return path
    
    def _compute_path_cost(
        self,
        path: torch.Tensor,
        gr_field: torch.Tensor,
        affordance: torch.Tensor
    ) -> torch.Tensor:
        """Compute cost of path through curved spacetime."""
        
        B, T, _ = path.shape
        device = path.device
        
        # Arc length in curved space: ds² = g_ij dx^i dx^j
        total_cost = torch.zeros(B, device=device)
        
        for t in range(T - 1):
            # Path segment
            p1 = path[:, t]
            p2 = path[:, t + 1]
            dp = p2 - p1
            
            # Sample metric at midpoint
            midpoint = (p1 + p2) / 2
            metric_local = self._sample_metric_at_point(
                gr_field, midpoint
            )
            
            # Compute interval
            # ds² = g_ij dp^i dp^j
            # For simplicity, using diagonal metric components
            g11 = metric_local[:, 4]  # g_xx
            g22 = metric_local[:, 7]  # g_yy
            g33 = metric_local[:, 9]  # g_zz
            
            ds_squared = (g11 * dp[:, 0] ** 2 +
                         g22 * dp[:, 1] ** 2 +
                         g33 * dp[:, 2] ** 2)
            
            ds = torch.sqrt(torch.abs(ds_squared) + 1e-8)
            
            # Add affordance penalty
            affordance_local = self._sample_affordance_at_point(
                affordance, midpoint
            )
            penalty = 1.0 + affordance_local.mean(dim=-1)
            
            total_cost += ds * penalty
            
        return total_cost
    
    def _sample_metric_at_point(
        self,
        gr_field: torch.Tensor,
        point: torch.Tensor
    ) -> torch.Tensor:
        """Sample metric tensor at given point."""
        
        B = point.shape[0]
        H, W = gr_field.shape[1:3]
        
        # Convert to grid coordinates
        # Assume space spans [-10, 10]
        grid_point = (point[:, :2] + 10) / 20 * torch.tensor([H, W], device=point.device)
        grid_point = grid_point.long()
        
        # Clamp to valid range
        grid_point[:, 0] = torch.clamp(grid_point[:, 0], 0, H - 1)
        grid_point[:, 1] = torch.clamp(grid_point[:, 1], 0, W - 1)
        
        # Sample
        sampled = []
        for b in range(B):
            i, j = grid_point[b]
            sampled.append(gr_field[b, i, j])
            
        return torch.stack(sampled)
    
    def _sample_affordance_at_point(
        self,
        affordance: torch.Tensor,
        point: torch.Tensor
    ) -> torch.Tensor:
        """Sample affordance at given point."""
        
        B = point.shape[0]
        H, W = affordance.shape[1:3]
        
        # Convert to grid coordinates
        grid_point = (point[:, :2] + 10) / 20 * torch.tensor([H, W], device=point.device)
        grid_point = grid_point.long()
        
        # Clamp
        grid_point[:, 0] = torch.clamp(grid_point[:, 0], 0, H - 1)
        grid_point[:, 1] = torch.clamp(grid_point[:, 1], 0, W - 1)
        
        # Sample
        sampled = []
        for b in range(B):
            i, j = grid_point[b]
            sampled.append(affordance[b, i, j])
            
        return torch.stack(sampled)
    
    def _compute_velocity_profile(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Compute velocity profile along trajectory."""
        
        B, T, _ = trajectory.shape
        
        # Finite differences
        velocity = torch.zeros_like(trajectory)
        velocity[:, 1:] = (trajectory[:, 1:] - trajectory[:, :-1]) / self.dt
        velocity[:, 0] = velocity[:, 1]  # Copy first velocity
        
        return velocity
    
    def _extract_waypoints(
        self,
        trajectory: torch.Tensor,
        num_waypoints: int
    ) -> torch.Tensor:
        """Extract key waypoints from trajectory."""
        
        B, T, D = trajectory.shape
        
        # Uniform sampling
        indices = torch.linspace(0, T - 1, num_waypoints).long()
        waypoints = trajectory[:, indices]
        
        return waypoints


class GeodesicSolver(nn.Module):
    """Solve geodesic equations for optimal paths."""
    
    def __init__(self, num_samples: int, max_iterations: int):
        super().__init__()
        
        self.num_samples = num_samples
        self.max_iterations = max_iterations
        
        # Learnable solver parameters
        self.step_size = nn.Parameter(torch.tensor(0.01))
        self.momentum = nn.Parameter(torch.tensor(0.9))
        
    def forward(
        self,
        path: torch.Tensor,
        metric_field: torch.Tensor,
        affordance: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve geodesic equation:
        d²x^μ/dτ² + Γ^μ_αβ (dx^α/dτ)(dx^β/dτ) = 0
        """
        
        B, T, D = path.shape
        optimized_path = path.clone()
        
        # Gradient descent on path length functional
        velocity = torch.zeros_like(path)
        
        for iteration in range(self.max_iterations):
            # Compute gradient of path length
            gradient = self._compute_path_gradient(
                optimized_path,
                metric_field,
                affordance
            )
            
            # Update with momentum
            velocity = self.momentum * velocity - self.step_size * gradient
            optimized_path = optimized_path + velocity
            
            # Keep endpoints fixed
            optimized_path[:, 0] = path[:, 0]
            optimized_path[:, -1] = path[:, -1]
            
        return optimized_path
    
    def _compute_path_gradient(
        self,
        path: torch.Tensor,
        metric_field: torch.Tensor,
        affordance: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient of path length functional."""
        
        B, T, D = path.shape
        gradient = torch.zeros_like(path)
        
        # Finite difference approximation
        eps = 1e-4
        
        for t in range(1, T - 1):
            for d in range(D):
                # Perturb path
                path_plus = path.clone()
                path_plus[:, t, d] += eps
                
                path_minus = path.clone()
                path_minus[:, t, d] -= eps
                
                # Compute costs
                cost_plus = self._local_path_cost(
                    path_plus[:, t-1:t+2],
                    metric_field,
                    affordance
                )
                
                cost_minus = self._local_path_cost(
                    path_minus[:, t-1:t+2],
                    metric_field,
                    affordance
                )
                
                # Gradient
                gradient[:, t, d] = (cost_plus - cost_minus) / (2 * eps)
                
        return gradient
    
    def _local_path_cost(
        self,
        path_segment: torch.Tensor,
        metric_field: torch.Tensor,
        affordance: torch.Tensor
    ) -> torch.Tensor:
        """Compute cost of local path segment."""
        
        # Simplified cost - would use full metric in practice
        distances = torch.norm(path_segment[:, 1:] - path_segment[:, :-1], dim=-1)
        return distances.sum(dim=-1)


class GoalEncoder(nn.Module):
    """Encode goal position from language features."""
    
    def __init__(self, language_dim: int, output_dim: int = 3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(language_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        
        # Goal position bounds
        self.position_scale = 10.0  # meters
        
    def forward(self, language_features: torch.Tensor) -> torch.Tensor:
        """Extract goal position from language."""
        
        # Pool language features
        if language_features.dim() == 3:
            pooled = language_features.mean(dim=1)
        else:
            pooled = language_features
            
        # Encode to position
        goal_position = self.encoder(pooled)
        
        # Scale to world coordinates
        goal_position = torch.tanh(goal_position) * self.position_scale
        
        return goal_position


class TrajectoryRefinementNetwork(nn.Module):
    """Neural network for trajectory refinement."""
    
    def __init__(self, hidden_dim: int, num_waypoints: int):
        super().__init__()
        
        self.num_waypoints = num_waypoints
        
        # Trajectory encoder
        self.trajectory_encoder = nn.LSTM(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Field encoder
        self.field_encoder = nn.Sequential(
            nn.Conv2d(10, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Refinement network
        self.refiner = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_waypoints * 3)
        )
        
    def forward(
        self,
        trajectory: torch.Tensor,
        gr_field: torch.Tensor,
        affordance: torch.Tensor,
        goal_features: torch.Tensor
    ) -> torch.Tensor:
        """Refine trajectory using learned corrections."""
        
        B, T, D = trajectory.shape
        
        # Encode trajectory
        traj_encoded, _ = self.trajectory_encoder(trajectory)
        traj_pooled = traj_encoded.mean(dim=1)  # [B, 2*hidden]
        
        # Encode field
        field_2d = gr_field.permute(0, 3, 1, 2)  # [B, 10, H, W]
        field_encoded = self.field_encoder(field_2d)  # [B, 64]
        
        # Combine features
        combined = torch.cat([traj_pooled, field_encoded], dim=-1)
        
        # Predict refinement
        refinement = self.refiner(combined)
        refinement = refinement.view(B, self.num_waypoints, 3)
        
        # Interpolate refinement to full trajectory
        refinement_interp = F.interpolate(
            refinement.transpose(1, 2),  # [B, 3, num_waypoints]
            size=T,
            mode='linear',
            align_corners=True
        ).transpose(1, 2)  # [B, T, 3]
        
        # Apply refinement
        refined = trajectory + 0.1 * refinement_interp
        
        # Keep endpoints fixed
        refined[:, 0] = trajectory[:, 0]
        refined[:, -1] = trajectory[:, -1]
        
        return refined


class PhysicsConstraints(nn.Module):
    """Enforce physical constraints on trajectories."""
    
    def __init__(self, max_vel: float, max_acc: float, dt: float):
        super().__init__()
        
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.dt = dt
        
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Apply velocity and acceleration constraints."""
        
        B, T, D = trajectory.shape
        constrained = trajectory.clone()
        
        # Iteratively enforce constraints
        for _ in range(3):
            # Compute velocities
            vel = (constrained[:, 1:] - constrained[:, :-1]) / self.dt
            
            # Limit velocities
            vel_norm = torch.norm(vel, dim=-1, keepdim=True)
            vel_scale = torch.minimum(
                torch.ones_like(vel_norm),
                self.max_vel / (vel_norm + 1e-8)
            )
            vel = vel * vel_scale
            
            # Reconstruct positions
            constrained[:, 1:] = constrained[:, :1] + torch.cumsum(vel * self.dt, dim=1)
            
            # Compute accelerations
            if T > 2:
                acc = (vel[:, 1:] - vel[:, :-1]) / self.dt
                
                # Limit accelerations
                acc_norm = torch.norm(acc, dim=-1, keepdim=True)
                acc_scale = torch.minimum(
                    torch.ones_like(acc_norm),
                    self.max_acc / (acc_norm + 1e-8)
                )
                acc = acc * acc_scale
                
                # Reconstruct velocities
                vel[:, 1:] = vel[:, :1] + torch.cumsum(acc * self.dt, dim=1)
                
                # Reconstruct positions again
                constrained[:, 1:] = constrained[:, :1] + torch.cumsum(vel * self.dt, dim=1)
                
        return constrained


class CollisionChecker(nn.Module):
    """Check for collisions along trajectory."""
    
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius
        
    def forward(
        self,
        trajectory: torch.Tensor,
        affordance_map: torch.Tensor
    ) -> torch.Tensor:
        """Check if trajectory is collision-free."""
        
        B, T, _ = trajectory.shape
        H, W = affordance_map.shape[1:3]
        
        valid = torch.ones(B, dtype=torch.bool, device=trajectory.device)
        
        for t in range(T):
            point = trajectory[:, t, :2]  # Use only x, y
            
            # Convert to grid coordinates
            grid_point = (point + 10) / 20 * torch.tensor([H, W], device=point.device)
            grid_point = grid_point.long()
            
            # Check bounds
            in_bounds = ((grid_point[:, 0] >= 0) & (grid_point[:, 0] < H) &
                        (grid_point[:, 1] >= 0) & (grid_point[:, 1] < W))
            
            valid = valid & in_bounds
            
            # Check affordance (high mass = obstacle)
            for b in range(B):
                if valid[b] and in_bounds[b]:
                    i, j = grid_point[b]
                    mass = affordance_map[b, i, j].sum()
                    if mass > 100:  # Threshold for obstacle
                        valid[b] = False
                        
        return valid


class PathScoringNetwork(nn.Module):
    """Score path quality using learned criteria."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, path_features: torch.Tensor) -> torch.Tensor:
        """Score path quality [0, 1]."""
        return self.scorer(path_features)

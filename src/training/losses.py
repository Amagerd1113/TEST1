"""
Loss functions for VLA-GR training.
Combines multiple objectives for end-to-end learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VLAGRLoss(nn.Module):
    """Combined loss function for VLA-GR framework."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.weights = config['training']['losses']
        
        # Individual loss components
        self.action_loss = ActionLoss()
        self.field_loss = FieldConsistencyLoss()
        self.affordance_loss = AffordancePredictionLoss()
        self.depth_loss = DepthCompletionLoss()
        self.entropy_loss = EntropyRegularizationLoss()
        self.path_loss = PathOptimalityLoss()
        
        # Additional regularization
        self.smoothness_loss = SmoothnessLoss()
        self.physics_loss = PhysicsViolationLoss(config)
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and individual components.
        
        Args:
            outputs: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of loss values
        """
        
        losses = {}
        
        # Action prediction loss
        if 'actions' in outputs and 'target_actions' in targets:
            losses['action'] = self.action_loss(
                outputs['actions'],
                targets['target_actions']
            )
            
        # GR field consistency loss
        if 'gr_field' in outputs:
            losses['field'] = self.field_loss(
                outputs['gr_field'],
                outputs.get('affordance_map')
            )
            
        # Affordance prediction loss
        if 'affordance_map' in outputs and 'target_affordance' in targets:
            losses['affordance'] = self.affordance_loss(
                outputs['affordance_map'],
                targets['target_affordance']
            )
            
        # Depth completion loss
        if 'completed_depth' in outputs and 'target_depth' in targets:
            losses['depth'] = self.depth_loss(
                outputs.get('completed_depth'),
                targets['target_depth'],
                outputs.get('occlusion_mask')
            )
            
        # Path optimality loss
        if 'planned_path' in outputs and 'optimal_path' in targets:
            losses['path'] = self.path_loss(
                outputs['planned_path'],
                targets['optimal_path'],
                outputs.get('gr_field')
            )
            
        # Entropy regularization
        if 'confidence' in outputs:
            losses['entropy'] = self.entropy_loss(outputs['confidence'])
            
        # Smoothness regularization
        if 'gr_field' in outputs:
            losses['smoothness'] = self.smoothness_loss(outputs['gr_field'])
            
        # Physics violation penalty
        if 'planned_path' in outputs:
            losses['physics'] = self.physics_loss(
                outputs['planned_path'],
                outputs.get('velocity_profile')
            )
            
        # Compute weighted total loss
        total_loss = torch.zeros(1, device=next(iter(outputs.values())).device)
        
        for key, weight in self.weights.items():
            if key in losses:
                total_loss += weight * losses[key]
                
        losses['total'] = total_loss
        
        return losses


class ActionLoss(nn.Module):
    """Loss for action prediction."""
    
    def __init__(self, loss_type: str = 'l2'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute action prediction loss."""
        
        if self.loss_type == 'l2':
            loss = F.mse_loss(pred_actions, target_actions)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(pred_actions, target_actions)
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred_actions, target_actions)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss


class FieldConsistencyLoss(nn.Module):
    """Loss for GR field physical consistency."""
    
    def forward(
        self,
        gr_field: torch.Tensor,
        affordance_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Ensure GR field satisfies physical constraints.
        
        - Metric tensor should be symmetric
        - Determinant should be negative (Lorentzian signature)
        - Field should be smooth
        """
        
        B, H, W, D = gr_field.shape
        loss = torch.zeros(1, device=gr_field.device)
        
        # Reconstruct full metric tensor from components
        # Components stored as: g_00, g_01, g_02, g_03, g_11, g_12, g_13, g_22, g_23, g_33
        metric = self._reconstruct_metric(gr_field)
        
        # Symmetry loss (should be automatic, but enforce numerically)
        symmetry_loss = torch.mean(
            (metric - metric.transpose(-2, -1)) ** 2
        )
        loss += 0.1 * symmetry_loss
        
        # Determinant constraint (should be negative for Lorentzian)
        det = torch.det(metric)
        det_loss = F.relu(det + 0.1)  # Penalize if det > -0.1
        loss += 0.01 * det_loss.mean()
        
        # Smoothness constraint
        if H > 1 and W > 1:
            # Spatial gradients
            dx = gr_field[:, 1:, :, :] - gr_field[:, :-1, :, :]
            dy = gr_field[:, :, 1:, :] - gr_field[:, :, :-1, :]
            
            smoothness = torch.mean(dx ** 2) + torch.mean(dy ** 2)
            loss += 0.01 * smoothness
            
        # Consistency with affordance
        if affordance_map is not None:
            # High affordance (obstacles) should curve spacetime more
            affordance_sum = affordance_map.sum(dim=-1)  # [B, H, W]
            
            # Compute field magnitude
            field_magnitude = torch.norm(gr_field, dim=-1)  # [B, H, W]
            
            # Correlation loss
            correlation = F.mse_loss(
                field_magnitude,
                torch.log1p(affordance_sum)
            )
            loss += 0.1 * correlation
            
        return loss
    
    def _reconstruct_metric(self, components: torch.Tensor) -> torch.Tensor:
        """Reconstruct 4x4 metric tensor from stored components."""
        B, H, W, _ = components.shape
        metric = torch.zeros(B, H, W, 4, 4, device=components.device)
        
        # Fill symmetric matrix
        # Diagonal elements
        metric[..., 0, 0] = components[..., 0]  # g_00
        metric[..., 1, 1] = components[..., 4]  # g_11
        metric[..., 2, 2] = components[..., 7]  # g_22
        metric[..., 3, 3] = components[..., 9]  # g_33
        
        # Off-diagonal elements (symmetric)
        metric[..., 0, 1] = metric[..., 1, 0] = components[..., 1]  # g_01
        metric[..., 0, 2] = metric[..., 2, 0] = components[..., 2]  # g_02
        metric[..., 0, 3] = metric[..., 3, 0] = components[..., 3]  # g_03
        metric[..., 1, 2] = metric[..., 2, 1] = components[..., 5]  # g_12
        metric[..., 1, 3] = metric[..., 3, 1] = components[..., 6]  # g_13
        metric[..., 2, 3] = metric[..., 3, 2] = components[..., 8]  # g_23
        
        return metric


class AffordancePredictionLoss(nn.Module):
    """Loss for affordance map prediction."""
    
    def forward(
        self,
        pred_affordance: torch.Tensor,
        target_affordance: torch.Tensor
    ) -> torch.Tensor:
        """Compute affordance prediction loss."""
        
        # L2 loss on affordance distributions
        loss = F.mse_loss(pred_affordance, target_affordance)
        
        # KL divergence for distribution matching
        if pred_affordance.shape[-1] > 1:
            pred_probs = F.softmax(pred_affordance, dim=-1)
            target_probs = F.softmax(target_affordance, dim=-1)
            
            kl_loss = F.kl_div(
                torch.log(pred_probs + 1e-8),
                target_probs,
                reduction='batchmean'
            )
            loss += 0.1 * kl_loss
            
        return loss


class DepthCompletionLoss(nn.Module):
    """Loss for depth completion in occluded regions."""
    
    def forward(
        self,
        pred_depth: Optional[torch.Tensor],
        target_depth: torch.Tensor,
        occlusion_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute depth completion loss."""
        
        if pred_depth is None:
            return torch.zeros(1, device=target_depth.device)
            
        # Base L1 loss
        loss = F.l1_loss(pred_depth, target_depth, reduction='none')
        
        # Weight by occlusion mask if available
        if occlusion_mask is not None:
            # Higher weight for occluded regions
            weights = 1.0 + 4.0 * occlusion_mask
            loss = loss * weights
            
        # Add gradient matching loss for smoothness
        if pred_depth.shape[-2] > 1 and pred_depth.shape[-1] > 1:
            pred_dx = torch.gradient(pred_depth, dim=-1)[0]
            pred_dy = torch.gradient(pred_depth, dim=-2)[0]
            
            target_dx = torch.gradient(target_depth, dim=-1)[0]
            target_dy = torch.gradient(target_depth, dim=-2)[0]
            
            grad_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
            loss = loss.mean() + 0.1 * grad_loss
        else:
            loss = loss.mean()
            
        return loss


class EntropyRegularizationLoss(nn.Module):
    """Entropy regularization for exploration."""
    
    def forward(self, confidence: torch.Tensor) -> torch.Tensor:
        """Encourage exploration through entropy maximization."""
        
        # Convert confidence to probability distribution
        probs = torch.stack([confidence, 1 - confidence], dim=-1)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        # Maximize entropy (minimize negative entropy)
        loss = -entropy.mean()
        
        return loss


class PathOptimalityLoss(nn.Module):
    """Loss for path optimality in curved spacetime."""
    
    def forward(
        self,
        pred_path: torch.Tensor,
        optimal_path: torch.Tensor,
        gr_field: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute path optimality loss."""
        
        # Trajectory matching loss
        traj_loss = F.mse_loss(pred_path, optimal_path)
        
        # Geodesic loss if GR field available
        if gr_field is not None:
            geodesic_loss = self._compute_geodesic_loss(pred_path, gr_field)
            traj_loss += 0.1 * geodesic_loss
            
        return traj_loss
    
    def _compute_geodesic_loss(
        self,
        path: torch.Tensor,
        gr_field: torch.Tensor
    ) -> torch.Tensor:
        """Compute deviation from geodesic."""
        
        B, T, D = path.shape
        
        # Compute path acceleration
        velocity = path[:, 1:] - path[:, :-1]
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        
        # Geodesic equation residual (simplified)
        # Should be zero for perfect geodesic
        residual = torch.norm(acceleration, dim=-1)
        
        return residual.mean()


class SmoothnessLoss(nn.Module):
    """Spatial smoothness regularization."""
    
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Compute smoothness loss for fields."""
        
        # Total variation loss
        if field.dim() == 4:  # [B, H, W, C]
            if field.shape[1] > 1 and field.shape[2] > 1:
                dx = torch.abs(field[:, 1:, :, :] - field[:, :-1, :, :])
                dy = torch.abs(field[:, :, 1:, :] - field[:, :, :-1, :])
                tv_loss = dx.mean() + dy.mean()
            else:
                tv_loss = torch.zeros(1, device=field.device)
        else:
            tv_loss = torch.zeros(1, device=field.device)
            
        return tv_loss


class PhysicsViolationLoss(nn.Module):
    """Penalize physics constraint violations."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.max_velocity = config['model']['path']['max_velocity']
        self.max_acceleration = config['model']['path']['max_acceleration']
        self.dt = config['model']['path']['dt']
        
    def forward(
        self,
        path: torch.Tensor,
        velocity_profile: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute physics violation penalties."""
        
        loss = torch.zeros(1, device=path.device)
        
        if velocity_profile is not None:
            # Velocity constraint violation
            vel_magnitude = torch.norm(velocity_profile, dim=-1)
            vel_violation = F.relu(vel_magnitude - self.max_velocity)
            loss += vel_violation.mean()
            
            # Acceleration constraint violation
            if velocity_profile.shape[1] > 1:
                acceleration = (velocity_profile[:, 1:] - velocity_profile[:, :-1]) / self.dt
                acc_magnitude = torch.norm(acceleration, dim=-1)
                acc_violation = F.relu(acc_magnitude - self.max_acceleration)
                loss += acc_violation.mean()
        else:
            # Compute from path
            if path.shape[1] > 1:
                velocity = (path[:, 1:] - path[:, :-1]) / self.dt
                vel_magnitude = torch.norm(velocity, dim=-1)
                vel_violation = F.relu(vel_magnitude - self.max_velocity)
                loss += vel_violation.mean()
                
                if path.shape[1] > 2:
                    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / self.dt
                    acc_magnitude = torch.norm(acceleration, dim=-1)
                    acc_violation = F.relu(acc_magnitude - self.max_acceleration)
                    loss += acc_violation.mean()
                    
        return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for vision-language alignment."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        visual_features: torch.Tensor,
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss between vision and language."""
        
        # Normalize features
        visual_norm = F.normalize(visual_features, p=2, dim=-1)
        language_norm = F.normalize(language_features, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(visual_norm, language_norm.t()) / self.temperature
        
        # Create labels (diagonal should be matched)
        B = similarity.shape[0]
        labels = torch.arange(B, device=similarity.device)
        
        # Cross-entropy loss both ways
        loss_v2l = F.cross_entropy(similarity, labels)
        loss_l2v = F.cross_entropy(similarity.t(), labels)
        
        return (loss_v2l + loss_l2v) / 2


class AuxiliaryTaskLoss(nn.Module):
    """Auxiliary task losses for better representation learning."""
    
    def __init__(self):
        super().__init__()
        
        self.semantic_loss = nn.CrossEntropyLoss()
        self.instance_loss = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary task losses."""
        
        losses = {}
        
        # Semantic segmentation
        if 'semantic_logits' in predictions and 'semantic_labels' in targets:
            losses['semantic'] = self.semantic_loss(
                predictions['semantic_logits'],
                targets['semantic_labels']
            )
            
        # Instance segmentation
        if 'instance_logits' in predictions and 'instance_labels' in targets:
            losses['instance'] = self.instance_loss(
                predictions['instance_logits'],
                targets['instance_labels'].float()
            )
            
        # Surface normal prediction
        if 'predicted_normals' in predictions and 'target_normals' in targets:
            losses['normals'] = F.cosine_similarity(
                predictions['predicted_normals'],
                targets['target_normals'],
                dim=-1
            ).mean()
            losses['normals'] = 1.0 - losses['normals']  # Convert to loss
            
        return losses

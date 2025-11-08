"""
VLA-GR Agent: Vision-Language-Action with General Relativity Navigation
Main agent orchestrating the complete navigation pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
from einops import rearrange, repeat

from .perception import PerceptionModule
from .affordance import AffordanceQuantifier
from .gr_field import GRFieldManager
from .path_optimizer import PathOptimizer
from .action_decoder import ActionDecoder


logger = logging.getLogger(__name__)


@dataclass
class VLAGRState:
    """Complete state representation for VLA-GR agent."""
    
    # Observations
    rgb_image: torch.Tensor  # [B, 3, H, W]
    depth_map: torch.Tensor  # [B, 1, H, W]
    language_instruction: List[str]
    
    # Derived representations
    visual_features: Optional[torch.Tensor] = None
    language_features: Optional[torch.Tensor] = None
    affordance_map: Optional[torch.Tensor] = None
    gr_field: Optional[torch.Tensor] = None
    planned_path: Optional[torch.Tensor] = None
    
    # Agent state
    position: Optional[torch.Tensor] = None  # [B, 3]
    orientation: Optional[torch.Tensor] = None  # [B, 4] quaternion
    velocity: Optional[torch.Tensor] = None  # [B, 3]
    
    # Metadata
    timestamp: float = 0.0
    episode_step: int = 0
    confidence: float = 1.0


class VLAGRAgent(nn.Module):
    """
    Main VLA-GR Agent implementing the complete navigation pipeline.
    
    Pipeline stages:
    1. Multimodal Perception: Process RGB-D + language inputs
    2. Affordance Quantification: Convert semantics to mass distributions
    3. GR Field Computation: Solve Einstein field equations
    4. Path Optimization: Compute geodesics in curved spacetime
    5. Action Execution: Generate robot control commands
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Initialize modules
        self.perception = PerceptionModule(config)
        self.affordance_quantifier = AffordanceQuantifier(config)
        self.gr_field_manager = GRFieldManager(config)
        self.path_optimizer = PathOptimizer(config)
        self.action_decoder = ActionDecoder(config)
        
        # VLA Transformer backbone
        self.vla_transformer = self._build_vla_transformer()
        
        # Field injection layers
        self.field_injection = FieldInjectionModule(
            hidden_dim=config['model']['vla']['hidden_dim'],
            field_dim=config['model']['gr_field']['field_dim']
        )
        
        # Exploration vs Exploitation
        self.exploration_module = EntropyBasedExploration(
            initial_epsilon=0.3,
            decay_rate=0.995,
            min_epsilon=0.05
        )
        
        # Memory and state tracking
        self.memory_bank = MemoryBank(
            capacity=1000,
            feature_dim=config['model']['vla']['hidden_dim']
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = UncertaintyEstimator(
            hidden_dim=config['model']['vla']['hidden_dim'],
            num_samples=10
        )
        
        # Initialize metrics
        self.metrics = {
            'success_rate': 0.0,
            'collision_rate': 0.0,
            'spl': 0.0,
            'trajectory_efficiency': 0.0
        }
        
    def _build_vla_transformer(self) -> nn.Module:
        """Build VLA transformer backbone."""
        config = self.config['model']['vla']
        
        return VLATransformer(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout=config['dropout'],
            action_dim=config['action_dim']
        )
        
    def forward(
        self,
        state: VLAGRState,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete VLA-GR pipeline.
        
        Args:
            state: Current VLA-GR state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary containing:
                - actions: Robot control commands
                - gr_field: Computed GR field
                - planned_path: Optimized trajectory
                - confidence: Action confidence scores
        """
        
        # Stage 1: Multimodal Perception
        perception_output = self.perception(
            rgb=state.rgb_image,
            depth=state.depth_map,
            language=state.language_instruction
        )
        state.visual_features = perception_output['visual_features']
        state.language_features = perception_output['language_features']
        
        # Handle occlusions if detected
        if perception_output['occlusion_mask'] is not None:
            completed_depth = perception_output['completed_depth']
            state.depth_map = completed_depth
            
        # Stage 2: Affordance Quantification
        affordance_output = self.affordance_quantifier(
            visual_features=state.visual_features,
            language_features=state.language_features,
            depth_map=state.depth_map
        )
        state.affordance_map = affordance_output['affordance_map']
        
        # Stage 3: GR Field Computation
        gr_field_output = self.gr_field_manager(
            affordance_map=state.affordance_map,
            position=state.position,
            velocity=state.velocity
        )
        state.gr_field = gr_field_output['metric_tensor']
        
        # Stage 4: Path Optimization
        path_output = self.path_optimizer(
            gr_field=state.gr_field,
            start_position=state.position,
            goal_features=state.language_features,
            affordance_map=state.affordance_map
        )
        state.planned_path = path_output['trajectory']
        
        # Stage 5: VLA Transformer with field injection
        vla_features = self._prepare_vla_input(state)
        
        # Inject GR field into attention mechanism
        vla_features = self.field_injection(
            features=vla_features,
            gr_field=state.gr_field
        )
        
        # Generate actions through transformer
        transformer_output = self.vla_transformer(vla_features)
        
        # Decode final actions
        actions = self.action_decoder(
            transformer_output=transformer_output,
            planned_path=state.planned_path,
            current_state=state
        )
        
        # Apply exploration strategy
        if not deterministic:
            actions = self.exploration_module(
                actions=actions,
                state=state,
                step=state.episode_step
            )
            
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator(
            features=transformer_output,
            actions=actions
        )
        
        # Update memory bank
        self.memory_bank.add(
            state=state,
            features=transformer_output,
            actions=actions
        )
        
        # Compile outputs
        outputs = {
            'actions': actions,
            'gr_field': state.gr_field,
            'planned_path': state.planned_path,
            'affordance_map': state.affordance_map,
            'confidence': 1.0 - uncertainty,
            'visual_features': state.visual_features,
            'language_features': state.language_features
        }
        
        # Log metrics
        self._update_metrics(state, outputs)
        
        return outputs
    
    def _prepare_vla_input(self, state: VLAGRState) -> torch.Tensor:
        """Prepare input features for VLA transformer."""
        
        # Concatenate all features
        features_list = []
        
        # Visual features
        if state.visual_features is not None:
            features_list.append(state.visual_features)
            
        # Language features
        if state.language_features is not None:
            # Expand language features to match visual spatial dimensions
            B, L, D = state.language_features.shape
            lang_expanded = repeat(
                state.language_features,
                'b l d -> b l h w d',
                h=1, w=1
            ).squeeze(2).squeeze(2)
            features_list.append(lang_expanded)
            
        # Affordance features
        if state.affordance_map is not None:
            aff_features = self._encode_affordance(state.affordance_map)
            features_list.append(aff_features)
            
        # Concatenate along feature dimension
        if features_list:
            combined_features = torch.cat(features_list, dim=-1)
        else:
            raise ValueError("No features available for VLA transformer")
            
        return combined_features
    
    def _encode_affordance(self, affordance_map: torch.Tensor) -> torch.Tensor:
        """Encode affordance map into feature representation."""
        # Simple encoding - can be made more sophisticated
        B, H, W, C = affordance_map.shape
        affordance_features = self.affordance_quantifier.encode(affordance_map)
        return affordance_features.view(B, -1, affordance_features.shape[-1])
    
    def _update_metrics(self, state: VLAGRState, outputs: Dict):
        """Update performance metrics."""
        # This would be connected to actual evaluation metrics
        pass
    
    def reset(self):
        """Reset agent state for new episode."""
        self.memory_bank.clear()
        self.exploration_module.reset()
        self.gr_field_manager.reset()
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.metrics = checkpoint['metrics']
        logger.info(f"Checkpoint loaded from {path}")


class VLATransformer(nn.Module):
    """VLA Transformer backbone for action generation."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        action_dim: int
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer."""
        
        for layer in self.layers:
            x = layer(x)
            
        # Pool over sequence dimension
        x = x.mean(dim=1)
        
        # Project to action space
        actions = self.output_projection(x)
        
        return actions


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and MLP."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block."""
        
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        
        # MLP with residual
        x = x + self.mlp(x)
        x = self.norm2(x)
        
        return x


class FieldInjectionModule(nn.Module):
    """Inject GR field information into feature representations."""
    
    def __init__(self, hidden_dim: int, field_dim: int):
        super().__init__()
        
        self.field_encoder = nn.Sequential(
            nn.Linear(field_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(
        self,
        features: torch.Tensor,
        gr_field: torch.Tensor
    ) -> torch.Tensor:
        """Inject field information into features."""
        
        # Encode field
        B, H, W, D = gr_field.shape
        field_flat = gr_field.view(B, -1, D)
        field_encoded = self.field_encoder(field_flat)
        
        # Align dimensions
        if features.dim() == 3 and field_encoded.dim() == 3:
            # Take spatial average of field for sequence features
            field_encoded = field_encoded.mean(dim=1, keepdim=True)
            field_encoded = field_encoded.expand(-1, features.shape[1], -1)
            
        # Fuse features
        fused = torch.cat([features, field_encoded], dim=-1)
        output = self.fusion(fused)
        
        return output


class EntropyBasedExploration(nn.Module):
    """Entropy-based exploration strategy."""
    
    def __init__(
        self,
        initial_epsilon: float = 0.3,
        decay_rate: float = 0.995,
        min_epsilon: float = 0.05
    ):
        super().__init__()
        self.initial_epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.current_epsilon = initial_epsilon
        
    def forward(
        self,
        actions: torch.Tensor,
        state: VLAGRState,
        step: int
    ) -> torch.Tensor:
        """Apply exploration noise to actions."""
        
        # Decay epsilon
        self.current_epsilon = max(
            self.min_epsilon,
            self.initial_epsilon * (self.decay_rate ** step)
        )
        
        # Add exploration noise
        if torch.rand(1).item() < self.current_epsilon:
            noise = torch.randn_like(actions) * 0.1
            actions = actions + noise
            
        return actions
    
    def reset(self):
        """Reset exploration parameters."""
        self.current_epsilon = self.initial_epsilon


class MemoryBank:
    """Memory bank for storing past experiences."""
    
    def __init__(self, capacity: int, feature_dim: int):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.memory = []
        self.pointer = 0
        
    def add(self, state: VLAGRState, features: torch.Tensor, actions: torch.Tensor):
        """Add experience to memory."""
        experience = {
            'state': state,
            'features': features.detach().cpu(),
            'actions': actions.detach().cpu(),
            'timestamp': state.timestamp
        }
        
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.pointer] = experience
            self.pointer = (self.pointer + 1) % self.capacity
            
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample batch from memory."""
        if len(self.memory) < batch_size:
            return self.memory
        
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]
    
    def clear(self):
        """Clear memory bank."""
        self.memory = []
        self.pointer = 0


class UncertaintyEstimator(nn.Module):
    """Estimate uncertainty in action predictions."""
    
    def __init__(self, hidden_dim: int, num_samples: int = 10):
        super().__init__()
        
        self.num_samples = num_samples
        
        # Ensemble of uncertainty heads
        self.uncertainty_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_samples)
        ])
        
    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Estimate uncertainty."""
        
        uncertainties = []
        
        for head in self.uncertainty_heads:
            unc = head(features)
            uncertainties.append(unc)
            
        # Average uncertainties
        uncertainty = torch.stack(uncertainties, dim=0).mean(dim=0)
        
        # Action-based uncertainty adjustment
        action_std = torch.std(actions, dim=-1, keepdim=True)
        uncertainty = uncertainty * (1 + action_std)
        
        return uncertainty.squeeze(-1)


class ActionDecoder(nn.Module):
    """Decode transformer output into robot actions."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        action_dim = config['model']['vla']['action_dim']
        hidden_dim = config['model']['vla']['hidden_dim']
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + action_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Action bounds
        self.register_buffer('action_low', torch.tensor([-1.0] * action_dim))
        self.register_buffer('action_high', torch.tensor([1.0] * action_dim))
        
    def forward(
        self,
        transformer_output: torch.Tensor,
        planned_path: torch.Tensor,
        current_state: VLAGRState
    ) -> torch.Tensor:
        """Decode actions from transformer output and planned path."""
        
        # Extract next waypoint from planned path
        if planned_path is not None and planned_path.shape[1] > 0:
            next_waypoint = planned_path[:, 0, :]  # [B, 3]
            
            # Compute direction to waypoint
            if current_state.position is not None:
                direction = next_waypoint - current_state.position
                direction = F.normalize(direction, p=2, dim=-1)
            else:
                direction = torch.zeros_like(next_waypoint)
                
            # Pad to action dimension
            direction_padded = F.pad(direction, (0, transformer_output.shape[-1] - 3))
        else:
            direction_padded = torch.zeros(
                transformer_output.shape[0],
                transformer_output.shape[-1],
                device=transformer_output.device
            )
            
        # Concatenate transformer output with planned direction
        if transformer_output.dim() == 3:
            transformer_output = transformer_output.mean(dim=1)
            
        features = torch.cat([
            transformer_output,
            direction_padded,
            direction_padded  # Duplicate for robustness
        ], dim=-1)
        
        # Decode actions
        actions = self.decoder(features)
        
        # Apply tanh and scale to action bounds
        actions = torch.tanh(actions)
        actions = self.action_low + (self.action_high - self.action_low) * (actions + 1) / 2
        
        return actions

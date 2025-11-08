"""
Affordance Quantification Module: Convert semantic understanding to Gaussian mass distributions.
Maps object properties to physical affordances for GR field computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)


class AffordanceQuantifier(nn.Module):
    """
    Convert semantic features to affordance mass distributions.
    
    Key concepts:
    - Semantic categories → Physical properties (mass, friction, traversability)
    - Gaussian distributions model uncertainty in affordance estimation
    - Bayesian updates incorporate environmental feedback
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Affordance parameters
        self.num_classes = config['model']['affordance']['num_classes']
        self.sigma_min = config['model']['affordance']['sigma_min']
        self.sigma_max = config['model']['affordance']['sigma_max']
        self.confidence_threshold = config['model']['affordance']['confidence_threshold']
        self.use_bayesian = config['model']['affordance']['use_bayesian_update']
        
        # Semantic to affordance mapping network
        self.semantic_encoder = SemanticEncoder(
            input_dim=config['model']['vla']['hidden_dim'],
            hidden_dim=512,
            num_classes=self.num_classes
        )
        
        # Affordance prediction heads
        self.mass_head = AffordanceHead(512, output_dim=1, activation='softplus')
        self.friction_head = AffordanceHead(512, output_dim=1, activation='sigmoid')
        self.traversability_head = AffordanceHead(512, output_dim=1, activation='sigmoid')
        self.uncertainty_head = AffordanceHead(512, output_dim=1, activation='sigmoid')
        
        # Gaussian parameters prediction
        self.gaussian_params = GaussianParameterNetwork(
            input_dim=512,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max
        )
        
        # Bayesian update module
        if self.use_bayesian:
            self.bayesian_updater = BayesianAffordanceUpdate(
                prior_strength=0.3,
                learning_rate=0.1
            )
            
        # Object property database (learned embeddings)
        self.object_properties = nn.Embedding(
            self.num_classes,
            256
        )
        
        # Spatial reasoning module
        self.spatial_reasoning = SpatialReasoningModule(
            hidden_dim=512,
            num_heads=8
        )
        
        # Initialize affordance statistics
        self._initialize_affordance_stats()
        
    def _initialize_affordance_stats(self):
        """Initialize default affordance statistics for common objects."""
        # Default mass values for common object categories (in kg)
        default_masses = {
            'floor': 1000.0,
            'wall': 500.0,
            'chair': 10.0,
            'table': 30.0,
            'person': 70.0,
            'door': 40.0,
            'plant': 5.0,
            'couch': 80.0,
            'bed': 100.0,
            'refrigerator': 150.0
        }
        
        # Convert to tensor
        self.register_buffer(
            'default_masses',
            torch.tensor([default_masses.get(f'class_{i}', 1.0) 
                         for i in range(self.num_classes)])
        )
        
    def forward(
        self,
        visual_features: torch.Tensor,
        language_features: torch.Tensor,
        depth_map: torch.Tensor,
        previous_affordance: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Quantify affordances from multimodal features.
        
        Args:
            visual_features: Encoded visual features [B, N, D]
            language_features: Encoded language features [B, L, D]
            depth_map: Depth information [B, 1, H, W]
            previous_affordance: Previous affordance map for Bayesian update
            
        Returns:
            Dictionary containing:
                - affordance_map: Gaussian mass distributions [B, H, W, C]
                - mass_map: Predicted mass values [B, H, W, 1]
                - friction_map: Surface friction coefficients [B, H, W, 1]
                - traversability_map: Traversability scores [B, H, W, 1]
                - uncertainty_map: Uncertainty estimates [B, H, W, 1]
        """
        
        B = visual_features.shape[0]
        device = visual_features.device
        
        # Encode semantic features
        semantic_features = self.semantic_encoder(
            visual_features,
            language_features
        )
        
        # Apply spatial reasoning
        semantic_features = self.spatial_reasoning(
            semantic_features,
            depth_map
        )
        
        # Predict affordance properties
        mass = self.mass_head(semantic_features)
        friction = self.friction_head(semantic_features)
        traversability = self.traversability_head(semantic_features)
        uncertainty = self.uncertainty_head(semantic_features)
        
        # Predict Gaussian parameters
        gaussian_params = self.gaussian_params(semantic_features)
        mu = gaussian_params['mu']
        sigma = gaussian_params['sigma']
        
        # Generate Gaussian mass distributions
        affordance_map = self._generate_gaussian_distributions(
            mu, sigma, mass, semantic_features
        )
        
        # Apply Bayesian update if available
        if self.use_bayesian and previous_affordance is not None:
            affordance_map = self.bayesian_updater(
                prior=previous_affordance,
                likelihood=affordance_map,
                confidence=1.0 - uncertainty
            )
            
        # Reshape to spatial dimensions
        H, W = depth_map.shape[-2:]
        N = semantic_features.shape[1]
        
        if N != H * W:
            # Interpolate to match spatial dimensions
            affordance_map = self._interpolate_to_spatial(
                affordance_map, H, W
            )
            mass = self._interpolate_to_spatial(mass, H, W)
            friction = self._interpolate_to_spatial(friction, H, W)
            traversability = self._interpolate_to_spatial(traversability, H, W)
            uncertainty = self._interpolate_to_spatial(uncertainty, H, W)
            
        return {
            'affordance_map': affordance_map,
            'mass_map': mass,
            'friction_map': friction,
            'traversability_map': traversability,
            'uncertainty_map': uncertainty,
            'gaussian_mu': mu,
            'gaussian_sigma': sigma
        }
    
    def _generate_gaussian_distributions(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        mass: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Generate Gaussian mass distributions for affordances."""
        
        B, N, D = features.shape
        device = features.device
        
        # Create spatial grid
        grid_size = int(np.sqrt(N))
        if grid_size * grid_size != N:
            # Handle non-square feature maps
            grid_size_h = int(np.sqrt(N * 4/3))  # Assume 4:3 aspect ratio
            grid_size_w = N // grid_size_h
        else:
            grid_size_h = grid_size_w = grid_size
            
        # Generate coordinate grid
        x = torch.linspace(-1, 1, grid_size_w, device=device)
        y = torch.linspace(-1, 1, grid_size_h, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        
        # Expand for batch
        coords = coords.unsqueeze(0).expand(B, -1, -1)
        
        # Compute Gaussian distributions
        # Each point contributes a Gaussian centered at mu with spread sigma
        affordance_distributions = []
        
        for i in range(N):
            # Get parameters for this spatial location
            mu_i = mu[:, i:i+1, :]  # [B, 1, 2]
            sigma_i = sigma[:, i:i+1, :]  # [B, 1, 2]
            mass_i = mass[:, i:i+1, :]  # [B, 1, 1]
            
            # Compute Gaussian weights
            diff = coords - mu_i  # [B, N, 2]
            
            # Anisotropic Gaussian
            weights = torch.exp(-0.5 * (
                (diff[..., 0] / sigma_i[..., 0]) ** 2 +
                (diff[..., 1] / sigma_i[..., 1]) ** 2
            ))  # [B, N]
            
            # Scale by mass
            weighted_distribution = weights * mass_i.squeeze(-1)
            affordance_distributions.append(weighted_distribution)
            
        # Stack all distributions
        affordance_map = torch.stack(affordance_distributions, dim=-1)  # [B, N, N]
        
        return affordance_map
    
    def _interpolate_to_spatial(
        self,
        features: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """Interpolate features to match spatial dimensions."""
        
        B, N, C = features.shape
        
        # Reshape to 2D grid
        grid_size = int(np.sqrt(N))
        if grid_size * grid_size == N:
            features_2d = features.reshape(B, grid_size, grid_size, C)
        else:
            # Handle non-square
            grid_h = int(np.sqrt(N * H / W))
            grid_w = N // grid_h
            features_2d = features[:, :grid_h*grid_w].reshape(B, grid_h, grid_w, C)
            
        # Permute for interpolation
        features_2d = features_2d.permute(0, 3, 1, 2)  # [B, C, H', W']
        
        # Interpolate
        features_interp = F.interpolate(
            features_2d,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        # Permute back
        features_interp = features_interp.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        return features_interp
    
    def encode(self, affordance_map: torch.Tensor) -> torch.Tensor:
        """Encode affordance map to feature representation."""
        B, H, W, C = affordance_map.shape
        
        # Flatten spatial dimensions
        affordance_flat = affordance_map.reshape(B, H * W, C)
        
        # Apply learned encoding
        encoded = F.gelu(F.linear(
            affordance_flat,
            weight=torch.randn(256, C, device=affordance_map.device) * 0.02
        ))
        
        return encoded


class SemanticEncoder(nn.Module):
    """Encode visual and language features to semantic representation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.language_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Semantic classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.output_projection = nn.Linear(hidden_dim * 2 + num_classes, hidden_dim)
        
    def forward(
        self,
        visual_features: torch.Tensor,
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode to semantic features."""
        
        # Encode modalities
        visual_encoded = self.visual_encoder(visual_features)
        language_encoded = self.language_encoder(language_features)
        
        # Cross-modal attention
        # Visual attending to language
        visual_attended, _ = self.cross_attention(
            query=visual_encoded,
            key=language_encoded,
            value=language_encoded
        )
        
        # Concatenate features
        combined = torch.cat([visual_encoded, visual_attended], dim=-1)
        
        # Predict semantic classes
        class_logits = self.classifier(combined)
        class_probs = F.softmax(class_logits, dim=-1)
        
        # Combine all features
        full_features = torch.cat([combined, class_probs], dim=-1)
        
        # Project to output dimension
        output = self.output_projection(full_features)
        
        return output


class AffordanceHead(nn.Module):
    """Prediction head for affordance properties."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 256,
        activation: str = 'sigmoid'
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Output activation
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict affordance property."""
        output = self.network(x)
        output = self.activation(output)
        return output


class GaussianParameterNetwork(nn.Module):
    """Predict Gaussian distribution parameters for affordances."""
    
    def __init__(
        self,
        input_dim: int,
        sigma_min: float = 0.1,
        sigma_max: float = 2.0
    ):
        super().__init__()
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # Mean prediction
        self.mu_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),  # 2D position
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        # Standard deviation prediction
        self.sigma_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),  # Anisotropic (different x, y)
            nn.Sigmoid()
        )
        
        # Correlation prediction (for full covariance)
        self.rho_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Correlation in [-1, 1]
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict Gaussian parameters."""
        
        # Predict mean
        mu = self.mu_net(x)
        
        # Predict standard deviation
        sigma_raw = self.sigma_net(x)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * sigma_raw
        
        # Predict correlation
        rho = self.rho_net(x)
        
        return {
            'mu': mu,
            'sigma': sigma,
            'rho': rho
        }


class BayesianAffordanceUpdate(nn.Module):
    """Bayesian update module for affordance refinement."""
    
    def __init__(
        self,
        prior_strength: float = 0.3,
        learning_rate: float = 0.1
    ):
        super().__init__()
        
        self.prior_strength = prior_strength
        self.learning_rate = learning_rate
        
        # Learnable combination weights
        self.combination_weights = nn.Parameter(
            torch.tensor([prior_strength, 1 - prior_strength])
        )
        
    def forward(
        self,
        prior: torch.Tensor,
        likelihood: torch.Tensor,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform Bayesian update on affordance distributions.
        
        Args:
            prior: Previous affordance estimate
            likelihood: New observation
            confidence: Confidence in new observation
            
        Returns:
            Updated affordance distribution
        """
        
        # Normalize weights
        weights = F.softmax(self.combination_weights, dim=0)
        
        # Weight by confidence
        likelihood_weight = weights[1] * confidence
        prior_weight = weights[0] + (1 - confidence) * weights[1]
        
        # Normalize weights
        total_weight = likelihood_weight + prior_weight
        likelihood_weight = likelihood_weight / (total_weight + 1e-8)
        prior_weight = prior_weight / (total_weight + 1e-8)
        
        # Combine distributions
        posterior = prior_weight.unsqueeze(-1) * prior + \
                   likelihood_weight.unsqueeze(-1) * likelihood
        
        return posterior


class SpatialReasoningModule(nn.Module):
    """Spatial reasoning for affordance relationships."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        # Spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Depth-aware position encoding
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, hidden_dim, 3, padding=1)
        )
        
        # Graph reasoning over spatial regions
        self.graph_conv = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        features: torch.Tensor,
        depth_map: torch.Tensor
    ) -> torch.Tensor:
        """Apply spatial reasoning to features."""
        
        B, N, D = features.shape
        
        # Extract depth-based position encoding
        depth_features = self.depth_encoder(depth_map)  # [B, D, H, W]
        depth_features_flat = depth_features.flatten(2).transpose(1, 2)  # [B, H*W, D]
        
        # Align dimensions
        if depth_features_flat.shape[1] != N:
            # Interpolate depth features
            depth_features_flat = F.interpolate(
                depth_features_flat.transpose(1, 2).unsqueeze(-1),
                size=(N, 1),
                mode='linear'
            ).squeeze(-1).transpose(1, 2)
            
        # Add depth-based position encoding
        features_with_depth = features + 0.1 * depth_features_flat
        
        # Apply spatial attention
        attended_features, attention_weights = self.spatial_attention(
            query=features_with_depth,
            key=features_with_depth,
            value=features_with_depth
        )
        
        # Graph reasoning
        # Create pairwise features
        features_expanded_1 = features.unsqueeze(2).expand(-1, -1, N, -1)
        features_expanded_2 = features.unsqueeze(1).expand(-1, N, -1, -1)
        pairwise_features = torch.cat([
            features_expanded_1,
            features_expanded_2
        ], dim=-1)  # [B, N, N, 2*D]
        
        # Apply graph convolution
        graph_features = self.graph_conv(pairwise_features)  # [B, N, N, D]
        
        # Aggregate graph features
        graph_aggregated = graph_features.mean(dim=2)  # [B, N, D]
        
        # Combine attended and graph features
        output = self.norm(attended_features + 0.5 * graph_aggregated)
        
        return output

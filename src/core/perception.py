"""
Perception Module: Multimodal perception with occlusion-aware depth completion.
Handles RGB-D inputs, language processing, and sensor fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class PerceptionModule(nn.Module):
    """
    Multimodal perception module for VLA-GR framework.
    
    Features:
    - RGB and depth image processing
    - Language instruction encoding
    - Occlusion detection and depth completion
    - Cross-modal attention mechanisms
    - Uncertainty-aware feature extraction
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Vision backbone
        self.vision_encoder = VisionEncoder(
            backbone=config['model']['vision']['backbone'],
            pretrained=config['model']['vision']['pretrained'],
            freeze_backbone=config['model']['vision']['freeze_backbone']
        )
        
        # Depth processing
        self.depth_encoder = DepthEncoder(
            in_channels=1,
            out_channels=256,
            use_completion=config['model']['occlusion']['completion_model'] is not None
        )
        
        # Depth completion network for occlusion handling
        if config['model']['occlusion']['completion_model'] == 'unet':
            self.depth_completion = UNetDepthCompletion(
                in_channels=4,  # RGB (3) + partial depth (1)
                out_channels=1
            )
        else:
            self.depth_completion = None
            
        # Language encoder
        self.language_encoder = LanguageEncoder(
            model_name=config['model']['language']['model'],
            max_tokens=config['model']['language']['max_tokens']
        )
        
        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusion(
            visual_dim=768,
            language_dim=config['model']['language']['embed_dim'],
            hidden_dim=config['model']['vla']['hidden_dim']
        )
        
        # Occlusion detection
        self.occlusion_detector = OcclusionDetector(
            threshold=config['model']['occlusion']['completion_threshold']
        )
        
        # Feature projection layers
        self.visual_projection = nn.Linear(768, config['model']['vla']['hidden_dim'])
        self.depth_projection = nn.Linear(256, config['model']['vla']['hidden_dim'])
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config['model']['vla']['hidden_dim'], 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        language: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through perception module.
        
        Args:
            rgb: RGB images [B, 3, H, W]
            depth: Depth maps [B, 1, H, W]
            language: List of language instructions
            
        Returns:
            Dictionary containing:
                - visual_features: Encoded visual features
                - language_features: Encoded language features
                - completed_depth: Completed depth map (if occlusions detected)
                - occlusion_mask: Binary mask of occluded regions
                - uncertainty: Perception uncertainty estimates
        """
        
        batch_size = rgb.shape[0]
        device = rgb.device
        
        # Detect occlusions in depth map
        occlusion_mask = self.occlusion_detector(depth)
        
        # Complete depth if occlusions detected
        completed_depth = depth
        if occlusion_mask is not None and self.depth_completion is not None:
            occlusion_ratio = occlusion_mask.float().mean()
            
            if occlusion_ratio > 0.05:  # More than 5% occluded
                logger.debug(f"Detected {occlusion_ratio:.2%} occlusion, completing depth")
                
                # Concatenate RGB and partial depth for completion
                completion_input = torch.cat([rgb, depth], dim=1)
                completed_depth = self.depth_completion(completion_input, occlusion_mask)
                
                # Blend completed and original depth
                completed_depth = depth * (1 - occlusion_mask.float()) + \
                                completed_depth * occlusion_mask.float()
        
        # Extract visual features
        visual_features = self.vision_encoder(rgb)  # [B, C, H', W']
        visual_features_flat = visual_features.flatten(2).transpose(1, 2)  # [B, H'*W', C]
        visual_features_proj = self.visual_projection(visual_features_flat)
        
        # Extract depth features
        depth_features = self.depth_encoder(completed_depth)  # [B, C, H', W']
        depth_features_flat = depth_features.flatten(2).transpose(1, 2)  # [B, H'*W', C]
        depth_features_proj = self.depth_projection(depth_features_flat)
        
        # Encode language instructions
        language_features = self.language_encoder(language, device=device)
        
        # Cross-modal fusion
        fused_features = self.cross_modal_fusion(
            visual_features_proj,
            depth_features_proj,
            language_features
        )
        
        # Estimate uncertainty
        pooled_features = fused_features.mean(dim=1)
        uncertainty = self.uncertainty_head(pooled_features)
        
        return {
            'visual_features': fused_features,
            'language_features': language_features,
            'completed_depth': completed_depth,
            'occlusion_mask': occlusion_mask,
            'uncertainty': uncertainty,
            'raw_visual_features': visual_features,
            'raw_depth_features': depth_features
        }


class VisionEncoder(nn.Module):
    """Vision backbone for RGB image encoding."""
    
    def __init__(
        self,
        backbone: str = 'dinov2_vitb14',
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        if backbone == 'dinov2_vitb14':
            # DINOv2 is particularly good for dense visual features
            self.backbone = self._load_dinov2(pretrained)
            self.feature_dim = 768
        elif backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=pretrained)
            # Remove final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
            
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Additional convolutional layers for feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(self.feature_dim, 768, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, 3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        
    def _load_dinov2(self, pretrained: bool) -> nn.Module:
        """Load DINOv2 model."""
        # Simplified DINOv2 loading - in practice would use actual pretrained weights
        class SimplifiedDINOv2(nn.Module):
            def __init__(self):
                super().__init__()
                # Simplified vision transformer
                self.patch_embed = nn.Conv2d(3, 768, kernel_size=14, stride=14)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=768,
                        nhead=12,
                        dim_feedforward=3072,
                        batch_first=True
                    ),
                    num_layers=12
                )
                
            def forward(self, x):
                # Patch embedding
                x = self.patch_embed(x)  # [B, 768, H/14, W/14]
                B, C, H, W = x.shape
                
                # Reshape for transformer
                x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
                
                # Apply transformer
                x = self.transformer(x)
                
                # Reshape back to spatial
                x = x.transpose(1, 2).reshape(B, C, H, W)
                
                return x
                
        return SimplifiedDINOv2()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual features from RGB images."""
        features = self.backbone(x)
        features = self.refine(features)
        return features


class DepthEncoder(nn.Module):
    """Depth map encoder with optional completion."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 256,
        use_completion: bool = True
    ):
        super().__init__()
        
        self.use_completion = use_completion
        
        # Depth encoding network
        self.encoder = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Residual blocks
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, out_channels, stride=1),
        )

        # Depth statistics extraction
        self.stats_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # min, max, mean depth
        )

        # Stats injection layer (initialized once, not in forward)
        self.stats_projection = nn.Conv2d(3, out_channels, 1)
        
    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Encode depth map to features."""
        
        # Normalize depth to [0, 1]
        depth_normalized = self._normalize_depth(depth)
        
        # Extract features
        features = self.encoder(depth_normalized)
        
        # Extract depth statistics
        stats = self.stats_extractor(features)

        # Inject statistics back into features
        B, C, H, W = features.shape
        stats_expanded = stats.unsqueeze(-1).unsqueeze(-1).expand(B, 3, H, W)
        stats_channels = self.stats_projection(stats_expanded)
        features = features + 0.1 * stats_channels

        return features
    
    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Normalize depth values to [0, 1] range."""
        # Handle invalid depth values
        valid_mask = (depth > 0) & (depth < 10.0)  # Assuming max depth of 10m
        
        if valid_mask.any():
            min_depth = depth[valid_mask].min()
            max_depth = depth[valid_mask].max()
            
            # Avoid division by zero
            if max_depth - min_depth > 1e-5:
                depth_normalized = (depth - min_depth) / (max_depth - min_depth)
            else:
                depth_normalized = torch.zeros_like(depth)
                
            # Set invalid depths to zero
            depth_normalized[~valid_mask] = 0
        else:
            depth_normalized = torch.zeros_like(depth)
            
        return depth_normalized


class ResidualBlock(nn.Module):
    """Residual block for depth encoder."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        
        return out


class UNetDepthCompletion(nn.Module):
    """U-Net for depth completion in occluded regions."""
    
    def __init__(self, in_channels: int = 4, out_channels: int = 1):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        
    def _conv_block(self, in_c: int, out_c: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Complete depth in occluded regions.
        
        Args:
            x: Input tensor (RGB + partial depth)
            mask: Binary mask of occluded regions
            
        Returns:
            Completed depth map
        """
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        # Output
        out = self.out_conv(d1)
        
        # Apply sigmoid to ensure positive depth values
        out = torch.sigmoid(out) * 10.0  # Scale to [0, 10] meters
        
        return out


class LanguageEncoder(nn.Module):
    """Language instruction encoder using pretrained LLM."""
    
    def __init__(self, model_name: str, max_tokens: int = 256):
        super().__init__()
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # Load pretrained model and tokenizer
        # Using a smaller model for efficiency
        if 'phi' in model_name.lower():
            # Microsoft Phi-2 is efficient and powerful
            # Phi-2 integrated in transformers>=4.37.0
            # Use AutoModelForCausalLM for best compatibility
            try:
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map="auto"  # Automatic device placement
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                # For feature extraction, we'll use the model's base
                # AutoModelForCausalLM has .model attribute with transformer layers
                logger.info(f"Loaded Phi-2 model: {model_name}")
            except Exception as e:
                # Fallback: use AutoModel for feature extraction
                logger.warning(f"AutoModelForCausalLM failed ({e}), using AutoModel")
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype="auto"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
        else:
            # Fallback to BERT for unknown models
            logger.info("Using BERT-base-uncased for language encoding")
            self.model = AutoModel.from_pretrained('bert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            
        # Freeze language model
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(
        self,
        instructions: List[str],
        device: torch.device
    ) -> torch.Tensor:
        """Encode language instructions to features."""
        
        # Tokenize instructions
        encoded = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True  # Ensure hidden states are returned
            )

        # Use last hidden states
        # For AutoModelForCausalLM, hidden_states is in outputs.hidden_states[-1]
        # For AutoModel, it's in outputs.last_hidden_state
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            features = outputs.hidden_states[-1]
        else:
            # Fallback: try to get from logits
            logger.warning("Could not find hidden states, using alternative extraction")
            features = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        return features


class CrossModalFusion(nn.Module):
    """Cross-modal attention fusion for vision and language."""
    
    def __init__(
        self,
        visual_dim: int,
        language_dim: int,
        hidden_dim: int
    ):
        super().__init__()
        
        # Project to common dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.depth_proj = nn.Linear(visual_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Cross-attention layers
        self.vision_language_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.language_vision_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(
        self,
        visual_features: torch.Tensor,
        depth_features: torch.Tensor,
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse multimodal features with cross-attention."""
        
        # Project features
        visual_proj = self.visual_proj(visual_features)
        depth_proj = self.depth_proj(depth_features)
        language_proj = self.language_proj(language_features)
        
        # Vision attending to language
        vision_attended, _ = self.vision_language_attention(
            query=visual_proj,
            key=language_proj,
            value=language_proj
        )
        
        # Language attending to vision
        language_attended, _ = self.language_vision_attention(
            query=language_proj,
            key=visual_proj,
            value=visual_proj
        )
        
        # Combine all features
        combined = torch.cat([
            vision_attended,
            depth_proj,
            language_attended.mean(dim=1, keepdim=True).expand_as(vision_attended)
        ], dim=-1)
        
        # Final fusion
        fused = self.fusion(combined)
        
        return fused


class OcclusionDetector(nn.Module):
    """Detect occluded regions in depth maps."""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

        # Learned occlusion detection
        self.detector = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, depth: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Detect occluded regions in depth map.

        Returns:
            Binary mask where 1 indicates occluded regions
        """

        # Simple heuristic: zero depth indicates occlusion
        occlusion_mask = (depth <= 0).float()

        # Learned detection for more complex occlusions
        learned_mask = self.detector(depth)

        # Combine heuristic and learned masks
        combined_mask = torch.max(occlusion_mask, learned_mask)

        # Threshold
        binary_mask = (combined_mask > self.threshold).float()

        # Return None if no occlusions
        if binary_mask.sum() == 0:
            return None

        return binary_mask


class AdvancedPerceptionModule(nn.Module):
    """
    Advanced perception module for ConferenceVLAGRAgent.
    Integrates all perception components with uncertainty estimation.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Use the main perception module
        self.perception = PerceptionModule(config)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        semantic: Optional[torch.Tensor] = None,
        language: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with advanced perception.

        Returns dict with:
            - visual_features: [B, N, D]
            - visual_uncertainty: [B, N, 1]
            - language_features: [B, L, D]
        """

        # Run base perception
        perception_output = self.perception(rgb, depth, language)

        # Expand uncertainty to per-token level
        B, N, D = perception_output['visual_features'].shape
        visual_uncertainty = perception_output['uncertainty'].unsqueeze(1).unsqueeze(1)
        visual_uncertainty = visual_uncertainty.expand(B, N, 1)

        return {
            'visual_features': perception_output['visual_features'],
            'visual_uncertainty': visual_uncertainty,
            'language_features': perception_output['language_features'],
            'completed_depth': perception_output['completed_depth'],
            'occlusion_mask': perception_output['occlusion_mask']
        }

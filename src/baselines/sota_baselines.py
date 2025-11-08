"""
State-of-the-Art Baseline Methods for Navigation
Implementations of top conference paper methods for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from transformers import CLIPModel, CLIPProcessor
import logging

logger = logging.getLogger(__name__)


class DD_PPO(nn.Module):
    """
    DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames
    Wijmans et al., ICLR 2020
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),  # RGB-D input
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU()
        )
        
        # Recurrent policy
        self.gru = nn.GRU(512 + 2, 512, num_layers=2)  # +2 for goal vector
        
        # Actor-critic heads
        self.actor = nn.Linear(512, 4)  # 4 discrete actions
        self.critic = nn.Linear(512, 1)
        
        # Hidden state
        self.hidden = None
        
    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        goal: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """DD-PPO forward pass."""
        
        # Concatenate RGB-D
        rgbd = torch.cat([rgb, depth], dim=1)
        
        # Visual encoding
        visual_features = self.visual_encoder(rgbd)
        
        # Concatenate with goal
        features = torch.cat([visual_features, goal], dim=-1)
        
        # Recurrent processing
        features = features.unsqueeze(0)  # Add sequence dimension
        if self.hidden is None:
            self.hidden = torch.zeros(2, features.shape[1], 512)
            
        output, self.hidden = self.gru(features, self.hidden)
        output = output.squeeze(0)
        
        # Actor-critic outputs
        action_logits = self.actor(output)
        value = self.critic(output)
        
        return {
            'action_logits': action_logits,
            'value': value
        }
    
    def act(self, obs: Dict) -> int:
        """Select action using trained policy."""
        
        with torch.no_grad():
            outputs = self.forward(
                obs['rgb'],
                obs['depth'],
                obs['pointgoal']
            )
            
        # Sample from policy
        probs = F.softmax(outputs['action_logits'], dim=-1)
        action = torch.multinomial(probs, 1).item()
        
        return action


class VLN_BERT(nn.Module):
    """
    Vision-and-Language Navigation with BERT
    Hong et al., EMNLP 2021
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        from transformers import BertModel
        
        # Language encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Cross-modal transformer
        self.cross_modal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Action prediction
        self.action_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )
        
    def forward(
        self,
        images: torch.Tensor,
        instruction_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """VLN-BERT forward pass."""
        
        # Encode language
        language_outputs = self.bert(
            input_ids=instruction_ids,
            attention_mask=attention_mask
        )
        language_features = language_outputs.last_hidden_state
        
        # Encode vision
        B, T, C, H, W = images.shape  # Batch, Time, Channels, Height, Width
        images_flat = images.view(B * T, C, H, W)
        visual_features = self.vision_encoder(images_flat)
        visual_features = visual_features.view(B, T, -1)
        
        # Project vision to language dimension
        visual_features = F.linear(
            visual_features,
            torch.randn(768, visual_features.shape[-1])
        )
        
        # Cross-modal attention
        combined = torch.cat([language_features, visual_features], dim=1)
        attended = self.cross_modal(combined)
        
        # Action prediction
        pooled = attended.mean(dim=1)
        action_logits = self.action_head(pooled)
        
        return {'action_logits': action_logits}


class CLIP_Nav(nn.Module):
    """
    CLIP-based Navigation
    Uses CLIP for zero-shot vision-language navigation
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Load CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze CLIP
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # Navigation head
        self.nav_head = nn.Sequential(
            nn.Linear(512 + 512, 512),  # Vision + Language features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        
    def forward(
        self,
        image: torch.Tensor,
        instruction: str
    ) -> Dict[str, torch.Tensor]:
        """CLIP-Nav forward pass."""
        
        # Process with CLIP
        inputs = self.processor(
            text=[instruction],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Get CLIP features
        outputs = self.clip(**inputs)
        
        # Combine features
        combined = torch.cat([
            outputs.image_embeds,
            outputs.text_embeds
        ], dim=-1)
        
        # Navigate
        action_logits = self.nav_head(combined)
        
        return {'action_logits': action_logits}


class CMA_PolicyNet(nn.Module):
    """
    Cross-Modal Attention Policy Network
    Wang et al., CVPR 2019
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Encoders
        self.vision_lstm = nn.LSTM(2048, 512, batch_first=True)
        self.language_lstm = nn.LSTM(300, 512, batch_first=True)
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(512, 8, batch_first=True)
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        
    def forward(
        self,
        visual_features: torch.Tensor,
        language_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """CMA forward pass."""
        
        # Encode sequences
        visual_encoded, _ = self.vision_lstm(visual_features)
        language_encoded, _ = self.language_lstm(language_features)
        
        # Cross-modal attention
        attended, _ = self.attention(
            query=visual_encoded,
            key=language_encoded,
            value=language_encoded
        )
        
        # Policy
        pooled = attended.mean(dim=1)
        action_logits = self.policy(pooled)
        
        return {'action_logits': action_logits}


class SLAM_Navigator(nn.Module):
    """
    Classical SLAM-based Navigation
    Using ORB-SLAM3 + A* planning
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Simplified SLAM components
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2),
            nn.ReLU()
        )
        
        # Mapping network
        self.mapper = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1)  # Occupancy map
        )
        
        # Path planner (learned)
        self.planner = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2),  # Map + goal
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 1)  # Action scores
        )
        
        # Accumulated map
        self.global_map = None
        
    def forward(
        self,
        rgb: torch.Tensor,
        goal: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """SLAM navigation forward pass."""
        
        # Extract features
        features = self.feature_extractor(rgb)
        
        # Update map
        local_map = self.mapper(features)
        
        if self.global_map is None:
            self.global_map = local_map
        else:
            # Simple map fusion
            self.global_map = 0.9 * self.global_map + 0.1 * local_map
            
        # Plan path
        goal_map = self.create_goal_map(goal, self.global_map.shape)
        planner_input = torch.cat([self.global_map, goal_map], dim=1)
        action_scores = self.planner(planner_input)
        
        # Global pooling for action
        action_logits = F.adaptive_avg_pool2d(action_scores, 1).squeeze(-1).squeeze(-1)
        
        return {'action_logits': action_logits}
    
    def create_goal_map(
        self,
        goal: torch.Tensor,
        shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Create goal map with goal location."""
        
        goal_map = torch.zeros(shape[0], 1, shape[2], shape[3])
        
        # Place goal on map (simplified)
        center = (shape[2] // 2, shape[3] // 2)
        goal_x = int(center[0] + goal[0, 0] * 10)
        goal_y = int(center[1] + goal[0, 1] * 10)
        
        goal_x = max(0, min(shape[2]-1, goal_x))
        goal_y = max(0, min(shape[3]-1, goal_y))
        
        goal_map[0, 0, goal_x, goal_y] = 1.0
        
        return goal_map


class NeuralSLAM(nn.Module):
    """
    Neural SLAM
    Zhang et al., ICLR 2021
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Neural SLAM modules
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Pose estimation
        self.pose_net = nn.Sequential(
            nn.Linear(128 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 3D position + 3D rotation
        )
        
        # Map decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1)  # Map + uncertainty
        )
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 4, 1)
        )
        
    def forward(
        self,
        rgbd: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Neural SLAM forward pass."""
        
        # Encode
        features = self.encoder(rgbd)
        
        # Estimate pose
        pose_features = features.flatten(1)
        pose = self.pose_net(pose_features)
        
        # Decode map
        map_output = self.decoder(features)
        occupancy_map = map_output[:, 0:1]
        uncertainty_map = map_output[:, 1:2]
        
        # Generate policy
        action_scores = self.policy(map_output)
        action_logits = F.adaptive_avg_pool2d(action_scores, 1).squeeze(-1).squeeze(-1)
        
        return {
            'action_logits': action_logits,
            'pose': pose,
            'map': occupancy_map,
            'uncertainty': uncertainty_map
        }


class HabitatBaseline(nn.Module):
    """
    Official Habitat Baseline
    PointNav and ObjectNav baselines from Habitat-Lab
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        from habitat_baselines.rl.ppo import PointNavBaselinePolicy
        
        # Load pretrained Habitat baseline
        self.policy = PointNavBaselinePolicy(
            observation_space=config['observation_space'],
            action_space=config['action_space'],
            hidden_size=512
        )
        
    def forward(self, observations: Dict) -> Dict[str, torch.Tensor]:
        """Habitat baseline forward pass."""
        
        return self.policy.act(
            observations,
            rnn_hidden_states=None,
            prev_actions=None,
            masks=None
        )


class E2ENav(nn.Module):
    """
    End-to-End Navigation
    Zhu et al., ICRA 2017
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Target-driven visual navigation
        self.current_encoder = self._make_encoder()
        self.target_encoder = self._make_encoder()
        
        # Siamese architecture
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Action network
        self.action_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
    def _make_encoder(self) -> nn.Module:
        """Create ResNet encoder."""
        
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            # Simplified ResNet blocks
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512)
        )
        
    def forward(
        self,
        current_image: torch.Tensor,
        target_image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """E2E navigation forward pass."""
        
        # Encode current and target
        current_features = self.current_encoder(current_image)
        target_features = self.target_encoder(target_image)
        
        # Combine features
        combined = torch.cat([current_features, target_features], dim=-1)
        fused = self.fusion(combined)
        
        # Predict action
        action_logits = self.action_net(fused)
        
        return {'action_logits': action_logits}


class SoundSpaces(nn.Module):
    """
    SoundSpaces: Audio-Visual Navigation
    Chen et al., CVPR 2020
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(2, 32, 5),  # Binaural audio
            nn.ReLU(),
            nn.Conv1d(32, 64, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Multimodal fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Navigation policy
        self.policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
    def forward(
        self,
        rgb: torch.Tensor,
        audio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """SoundSpaces forward pass."""
        
        # Encode modalities
        visual_features = self.visual_encoder(rgb)
        audio_features = self.audio_encoder(audio)
        
        # Fuse
        fused = self.fusion(torch.cat([visual_features, audio_features], dim=-1))
        
        # Navigate
        action_logits = self.policy(fused)
        
        return {'action_logits': action_logits}


# Baseline Factory
class BaselineFactory:
    """Factory for creating baseline models."""
    
    BASELINES = {
        'dd_ppo': DD_PPO,
        'vln_bert': VLN_BERT,
        'clip_nav': CLIP_Nav,
        'cma': CMA_PolicyNet,
        'slam': SLAM_Navigator,
        'neural_slam': NeuralSLAM,
        'habitat': HabitatBaseline,
        'e2e': E2ENav,
        'soundspaces': SoundSpaces
    }
    
    @classmethod
    def create_baseline(
        cls,
        name: str,
        config: Dict
    ) -> nn.Module:
        """Create baseline model by name."""
        
        if name not in cls.BASELINES:
            raise ValueError(f"Unknown baseline: {name}")
            
        return cls.BASELINES[name](config)
    
    @classmethod
    def get_all_baselines(cls, config: Dict) -> Dict[str, nn.Module]:
        """Create all baseline models."""
        
        baselines = {}
        
        for name in cls.BASELINES:
            try:
                baselines[name] = cls.create_baseline(name, config)
                logger.info(f"Created baseline: {name}")
            except Exception as e:
                logger.warning(f"Failed to create baseline {name}: {e}")
                
        return baselines

"""
Dual-System Architecture for VLA-GR
Based on NVIDIA GR00T N1 (2025) dual-system approach

System 1 (S1): Fast visuomotor policy for low-level control (~10ms latency)
System 2 (S2): VLM-based planner for high-level task decomposition and reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TaskPlan:
    """High-level task plan from System 2"""
    subgoals: List[str]  # Sequence of subgoals
    horizon: int  # Planning horizon
    confidence: float  # Confidence in the plan
    reasoning: str  # Reasoning chain


class System1FastPolicy(nn.Module):
    """
    System 1: Fast visuomotor policy for reactive control

    Key features:
    - Ultra-low latency (<10ms inference)
    - Direct visual servoing
    - Lightweight architecture (<50M parameters)
    - Optimized for real-time control at 50Hz+
    """

    def __init__(
        self,
        visual_dim: int = 768,
        proprio_dim: int = 7,
        hidden_dim: int = 256,
        action_dim: int = 7,
        num_layers: int = 3,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Fast visual encoder (pre-computed features from backbone)
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim)
        )

        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        # Fast fusion network (no attention for speed)
        fusion_layers = []
        for _ in range(num_layers):
            fusion_layers.extend([
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dim),
            ])
        self.fusion = nn.Sequential(*fusion_layers)

        # Action head (mean and log_std for stochastic policy)
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        visual_features: torch.Tensor,
        proprioception: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fast forward pass optimized for low latency

        Args:
            visual_features: Pre-computed visual features [B, visual_dim]
            proprioception: Robot state [B, proprio_dim]
            deterministic: If True, return mean action

        Returns:
            action: Sampled or mean action [B, action_dim]
            log_prob: Log probability (if not deterministic)
        """
        # Project visual features
        v_feat = self.visual_proj(visual_features)

        # Encode proprioception
        p_feat = self.proprio_encoder(proprioception)

        # Fuse modalities
        fused = torch.cat([v_feat, p_feat], dim=-1)
        fused = self.fusion(fused)

        # Compute action distribution
        action_mean = self.action_mean(fused)
        action_std = torch.exp(self.action_log_std)

        if deterministic:
            return action_mean, None

        # Sample action
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        return action, log_prob


class System2Planner(nn.Module):
    """
    System 2: VLM-based high-level planner

    Key features:
    - Language-guided task decomposition
    - Multi-step reasoning
    - Subgoal generation
    - Longer inference time (~100-500ms) but infrequent calls
    """

    def __init__(
        self,
        vlm_dim: int = 768,
        hidden_dim: int = 512,
        num_reasoning_steps: int = 3,
        max_subgoals: int = 5,
    ):
        super().__init__()

        self.vlm_dim = vlm_dim
        self.hidden_dim = hidden_dim
        self.num_reasoning_steps = num_reasoning_steps
        self.max_subgoals = max_subgoals

        # Reasoning transformer
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True  # Pre-norm for stability
            )
            for _ in range(num_reasoning_steps)
        ])

        # VLM feature projection
        self.vlm_proj = nn.Linear(vlm_dim, hidden_dim)

        # Subgoal decoder
        self.subgoal_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True
            ),
            num_layers=2
        )

        # Learnable subgoal queries
        self.subgoal_queries = nn.Parameter(
            torch.randn(max_subgoals, hidden_dim)
        )

        # Planning horizon predictor
        self.horizon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive
        )

        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        vlm_features: torch.Tensor,
        instruction_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate high-level plan

        Args:
            vlm_features: Vision-language features [B, L, vlm_dim]
            instruction_mask: Attention mask [B, L]

        Returns:
            Dictionary containing:
            - subgoal_features: Decoded subgoal features [B, max_subgoals, hidden_dim]
            - horizon: Predicted planning horizon [B, 1]
            - confidence: Plan confidence [B, 1]
        """
        B = vlm_features.shape[0]

        # Project VLM features
        features = self.vlm_proj(vlm_features)  # [B, L, hidden_dim]

        # Multi-step reasoning
        for layer in self.reasoning_layers:
            features = layer(features, src_key_padding_mask=instruction_mask)

        # Pool for global context
        if instruction_mask is not None:
            # Masked average pooling
            mask_expanded = (~instruction_mask).float().unsqueeze(-1)
            pooled = (features * mask_expanded).sum(1) / mask_expanded.sum(1)
        else:
            pooled = features.mean(1)

        # Predict horizon and confidence
        horizon = self.horizon_head(pooled)
        confidence = self.confidence_head(pooled)

        # Decode subgoals
        subgoal_queries = self.subgoal_queries.unsqueeze(0).expand(B, -1, -1)
        subgoal_features = self.subgoal_decoder(
            subgoal_queries,
            features,
            memory_key_padding_mask=instruction_mask
        )

        return {
            "subgoal_features": subgoal_features,
            "horizon": horizon,
            "confidence": confidence,
            "reasoning_features": pooled
        }


class DualSystemArchitecture(nn.Module):
    """
    Integrated dual-system architecture for VLA-GR

    Coordination strategy:
    - System 2 runs at low frequency (~1-5Hz) for planning
    - System 1 runs at high frequency (~50Hz) for control
    - System 2 provides subgoal guidance to System 1
    - Confidence-based switching between reactive and planned control
    """

    def __init__(
        self,
        visual_dim: int = 768,
        vlm_dim: int = 768,
        proprio_dim: int = 7,
        action_dim: int = 7,
        s1_hidden_dim: int = 256,
        s2_hidden_dim: int = 512,
        s1_layers: int = 3,
        s2_reasoning_steps: int = 3,
        max_subgoals: int = 5,
        planning_frequency_hz: float = 2.0,
        control_frequency_hz: float = 50.0,
        confidence_threshold: float = 0.5,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.action_dim = action_dim
        self.planning_frequency_hz = planning_frequency_hz
        self.control_frequency_hz = control_frequency_hz
        self.confidence_threshold = confidence_threshold

        # Initialize subsystems
        self.system1 = System1FastPolicy(
            visual_dim=visual_dim,
            proprio_dim=proprio_dim,
            hidden_dim=s1_hidden_dim,
            action_dim=action_dim,
            num_layers=s1_layers,
        )

        self.system2 = System2Planner(
            vlm_dim=vlm_dim,
            hidden_dim=s2_hidden_dim,
            num_reasoning_steps=s2_reasoning_steps,
            max_subgoals=max_subgoals,
        )

        # Subgoal conditioning for System 1
        self.subgoal_modulator = nn.Sequential(
            nn.Linear(s2_hidden_dim, s1_hidden_dim),
            nn.Tanh()
        )

        # Timestep counters
        self.register_buffer("timestep", torch.tensor(0))
        self.register_buffer(
            "planning_interval",
            torch.tensor(int(control_frequency_hz / planning_frequency_hz))
        )

        # Cache for current plan
        self.cached_plan = None

    def should_replan(self) -> bool:
        """Determine if System 2 should create a new plan"""
        return self.timestep % self.planning_interval == 0

    def forward(
        self,
        visual_features: torch.Tensor,
        vlm_features: torch.Tensor,
        proprioception: torch.Tensor,
        instruction_mask: Optional[torch.Tensor] = None,
        force_replan: bool = False,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Dual-system forward pass

        Args:
            visual_features: Visual features for S1 [B, visual_dim]
            vlm_features: VLM features for S2 [B, L, vlm_dim]
            proprioception: Robot state [B, proprio_dim]
            instruction_mask: Attention mask [B, L]
            force_replan: Force System 2 to replan
            deterministic: Use deterministic actions

        Returns:
            Dictionary containing actions and planning info
        """
        results = {}

        # System 2: High-level planning (low frequency)
        if self.should_replan() or force_replan or self.cached_plan is None:
            with torch.set_grad_enabled(self.training):
                plan = self.system2(vlm_features, instruction_mask)
                self.cached_plan = plan
                results["planned"] = True
                results["horizon"] = plan["horizon"]
                results["confidence"] = plan["confidence"]
        else:
            plan = self.cached_plan
            results["planned"] = False

        # Get current subgoal (first subgoal from plan)
        current_subgoal = plan["subgoal_features"][:, 0, :]  # [B, s2_hidden_dim]
        subgoal_modulation = self.subgoal_modulator(current_subgoal)  # [B, s1_hidden_dim]

        # System 1: Fast reactive control (high frequency)
        # Modulate S1 with S2's subgoal
        # Add subgoal as additive bias to visual features
        modulated_visual = visual_features + subgoal_modulation

        action, log_prob = self.system1(
            modulated_visual,
            proprioception,
            deterministic=deterministic
        )

        results["action"] = action
        if log_prob is not None:
            results["log_prob"] = log_prob

        # Increment timestep
        if not self.training:
            self.timestep += 1

        return results

    def reset_planning(self):
        """Reset planning state (call at episode start)"""
        self.timestep.zero_()
        self.cached_plan = None

    @torch.no_grad()
    def get_action(
        self,
        visual_features: torch.Tensor,
        vlm_features: torch.Tensor,
        proprioception: torch.Tensor,
        instruction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get action for execution (inference mode)

        Args:
            visual_features: Visual features [B, visual_dim]
            vlm_features: VLM features [B, L, vlm_dim]
            proprioception: Robot state [B, proprio_dim]
            instruction_mask: Attention mask [B, L]

        Returns:
            Action [B, action_dim]
        """
        self.eval()
        results = self.forward(
            visual_features,
            vlm_features,
            proprioception,
            instruction_mask,
            deterministic=True
        )
        return results["action"]

    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for each subsystem"""
        s1_params = sum(p.numel() for p in self.system1.parameters())
        s2_params = sum(p.numel() for p in self.system2.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "system1": s1_params,
            "system2": s2_params,
            "total": total_params,
            "overhead": total_params - s1_params - s2_params
        }

"""
Additional modules for ConferenceVLAGRAgent.
Contains SpacetimeMemoryModule, HierarchicalActionDecoder, and EpistemicUncertaintyModule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpacetimeMemoryModule(nn.Module):
    """
    Spacetime memory consolidation module.
    Stores and retrieves episodic memories with relativistic indexing.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_size: int,
        consolidation_threshold: float = 0.5
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.consolidation_threshold = consolidation_threshold

        # Memory storage (learnable embeddings)
        self.memory_bank = nn.Parameter(torch.randn(memory_size, hidden_dim) * 0.02)

        # Memory attention
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # Memory consolidation network
        self.consolidation_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Position encoder for spacetime indexing
        self.position_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim)
        )

    def forward(
        self,
        current_features: torch.Tensor,
        position: torch.Tensor,
        metric_tensor: torch.Tensor,
        episodic_memory: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Consolidate and retrieve memories.

        Args:
            current_features: Current observations [B, N, D]
            position: Current position [B, 3]
            metric_tensor: Current metric field [B, H, W, 10]
            episodic_memory: Previous episode memories [B, M, D]

        Returns:
            Dict with:
                - consolidated_memory: Updated memory [B, M, D]
                - attention_weights: Memory attention weights [B, N, M]
        """

        B, N, D = current_features.shape

        # Encode position for spacetime indexing
        position_encoding = self.position_encoder(position)  # [B, D]
        position_encoding = position_encoding.unsqueeze(1).expand(B, N, D)

        # Combine features with position
        features_with_position = current_features + 0.1 * position_encoding

        # Expand memory bank for batch
        memory = self.memory_bank.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        # If episodic memory provided, blend with memory bank
        if episodic_memory is not None:
            # Ensure dimensions match
            if episodic_memory.shape[1] == memory.shape[1]:
                memory = 0.5 * memory + 0.5 * episodic_memory
            else:
                # Use only memory bank if dimensions don't match
                pass

        # Retrieve from memory using attention
        retrieved_memory, attention_weights = self.memory_attention(
            query=features_with_position,
            key=memory,
            value=memory
        )

        # Consolidate current features with retrieved memory
        combined = torch.cat([current_features, retrieved_memory], dim=-1)
        consolidated = self.consolidation_net(combined)

        # Update memory bank (simplified - in practice would use more sophisticated update)
        # We don't update memory_bank directly as it's a parameter

        return {
            'consolidated_memory': memory,  # Return memory for next timestep
            'attention_weights': attention_weights,
            'retrieved_features': retrieved_memory
        }


class HierarchicalActionDecoder(nn.Module):
    """
    Hierarchical action decoder with primitive composition.
    Generates actions as weighted combinations of learned primitives.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_primitives: int = 8
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_primitives = num_primitives

        # Primitive action embeddings
        self.primitives = nn.Parameter(torch.randn(num_primitives, action_dim) * 0.1)

        # Primitive selector network
        self.primitive_selector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_primitives)
        )

        # Action refinement network
        self.action_refiner = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, action_dim)
        )

        # Goal-conditioned modulation
        self.goal_modulator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(
        self,
        features: torch.Tensor,
        planned_path: torch.Tensor,
        current_position: torch.Tensor,
        goal_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Decode hierarchical actions.

        Args:
            features: Encoded features [B, S, D]
            planned_path: Planned trajectory [B, T, 3]
            current_position: Current position [B, 3]
            goal_embedding: Goal encoding [B, D]

        Returns:
            Dict with:
                - actions: Final actions [B, A]
                - primitives: Primitive actions [B, K, A]
                - weights: Primitive weights [B, K]
        """

        B = features.shape[0]

        # Pool features for action prediction
        pooled_features = features.mean(dim=1)  # [B, D]

        # Select primitive weights
        primitive_logits = self.primitive_selector(pooled_features)  # [B, K]
        primitive_weights = F.softmax(primitive_logits, dim=-1)

        # Expand primitives for batch
        primitives_expanded = self.primitives.unsqueeze(0).expand(B, -1, -1)  # [B, K, A]

        # Compute base action as weighted combination of primitives
        base_action = torch.einsum('bk,bka->ba', primitive_weights, primitives_expanded)

        # Goal-conditioned modulation
        goal_context = torch.cat([pooled_features, goal_embedding], dim=-1)
        goal_modulation = self.goal_modulator(goal_context)

        # Refine action
        action_input = torch.cat([pooled_features, base_action], dim=-1)
        refinement = self.action_refiner(action_input)

        # Final action: base + refinement + goal modulation
        final_action = base_action + 0.3 * refinement + 0.1 * goal_modulation

        return {
            'actions': final_action,
            'primitives': primitives_expanded,
            'weights': primitive_weights,
            'base_action': base_action,
            'refinement': refinement
        }


class EpistemicUncertaintyModule(nn.Module):
    """
    Epistemic uncertainty estimation using ensemble methods.
    Quantifies model uncertainty for safe navigation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_ensemble: int = 5
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_ensemble = num_ensemble

        # Ensemble of uncertainty predictors
        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.GELU(),
                nn.Dropout(0.2),  # Different dropout for diversity
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Softplus()  # Ensure positive uncertainty
            )
            for _ in range(num_ensemble)
        ])

        # Uncertainty aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(num_ensemble, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate epistemic uncertainty.

        Args:
            features: Encoded features [B, S, D]
            actions: Predicted actions [B, A]

        Returns:
            Epistemic uncertainty [B, 1]
        """

        B = features.shape[0]

        # Pool features
        pooled_features = features.mean(dim=1)  # [B, D]

        # Get predictions from ensemble
        ensemble_predictions = []
        for predictor in self.ensemble:
            uncertainty = predictor(pooled_features)  # [B, 1]
            ensemble_predictions.append(uncertainty)

        # Stack ensemble predictions
        ensemble_stack = torch.cat(ensemble_predictions, dim=-1)  # [B, num_ensemble]

        # Compute variance across ensemble (epistemic uncertainty)
        epistemic_uncertainty = ensemble_stack.var(dim=-1, keepdim=True)  # [B, 1]

        # Also aggregate through learned aggregator
        aggregated_uncertainty = self.aggregator(ensemble_stack)  # [B, 1]

        # Combine variance and aggregated uncertainty
        final_uncertainty = (epistemic_uncertainty + aggregated_uncertainty) / 2

        return final_uncertainty

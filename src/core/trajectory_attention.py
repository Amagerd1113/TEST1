"""
Trajectory Attention Mechanism for VLA-GR
Based on Actra (2024) architecture improvements

Key innovations:
- Temporal-aware attention over action trajectories
- Learnable action queries for efficient trajectory encoding
- Causal masking for autoregressive action prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for better temporal modeling
    Used in modern transformers for superior position encoding
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys

        Args:
            q: Query tensor [B, num_heads, seq_len, head_dim]
            k: Key tensor [B, num_heads, seq_len, head_dim]

        Returns:
            Rotated q and k
        """
        seq_len = q.shape[2]

        # Get cached cos/sin
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]

        # Apply rotation
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed


class TrajectoryAttention(nn.Module):
    """
    Specialized attention mechanism for action trajectories

    Features:
    - Temporal-aware attention with RoPE
    - Efficient computation for long trajectories
    - Causal masking for autoregressive prediction
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_trajectory_len: int = 100,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Rotary position embedding
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim,
                max_seq_len=max_trajectory_len
            )

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_trajectory_len, max_trajectory_len) * float("-inf"),
                diagonal=1
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with trajectory attention

        Args:
            x: Input trajectory embeddings [B, T, hidden_dim]
            context: Optional context for cross-attention [B, L, hidden_dim]
            causal: Whether to use causal masking
            attention_mask: Optional attention mask [B, T] or [B, T, T]

        Returns:
            Attended features [B, T, hidden_dim]
        """
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x if context is None else context)
        v = self.v_proj(x if context is None else context)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope and context is None:  # Only for self-attention
            q, k = self.rope(q, k)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if causal and context is None:
            seq_len = attn_scores.size(-1)
            causal_mask = self.causal_mask[:seq_len, :seq_len]
            attn_scores = attn_scores + causal_mask

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [B, T] -> [B, 1, 1, T]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out


class LearnableActionQueries(nn.Module):
    """
    Learnable action queries for efficient trajectory encoding
    Inspired by DETR-style object queries
    """

    def __init__(
        self,
        num_queries: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))

        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(
        self,
        features: torch.Tensor,
        num_actions: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode actions from queries

        Args:
            features: Context features [B, L, hidden_dim]
            num_actions: Number of actions to predict (default: num_queries)

        Returns:
            queries: Query embeddings [B, num_actions, hidden_dim]
            actions: Predicted actions [B, num_actions, action_dim]
        """
        B = features.shape[0]
        if num_actions is None:
            num_actions = self.num_queries

        # Expand queries for batch
        queries = self.query_embed[:num_actions].unsqueeze(0).expand(B, -1, -1)

        # Decode actions
        actions = self.action_decoder(queries)

        return queries, actions


class TrajectoryTransformerBlock(nn.Module):
    """
    Transformer block optimized for trajectory processing

    Combines:
    - Trajectory self-attention
    - Cross-attention to visual/language context
    - Feed-forward network
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Self-attention for trajectories
        self.self_attn = TrajectoryAttention(
            hidden_dim,
            num_heads,
            dropout,
            use_rope=use_rope
        )

        # Cross-attention to context
        self.cross_attn = TrajectoryAttention(
            hidden_dim,
            num_heads,
            dropout,
            use_rope=False  # No RoPE for cross-attention
        )

        # Feed-forward network
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Trajectory embeddings [B, T, hidden_dim]
            context: Context features [B, L, hidden_dim]
            causal: Use causal masking for self-attention
            attention_mask: Attention mask

        Returns:
            Output features [B, T, hidden_dim]
        """
        # Self-attention
        x = x + self.dropout(self.self_attn(
            self.norm1(x),
            causal=causal,
            attention_mask=attention_mask
        ))

        # Cross-attention
        x = x + self.dropout(self.cross_attn(
            self.norm2(x),
            context=context,
            causal=False
        ))

        # FFN
        x = x + self.dropout(self.mlp(self.norm3(x)))

        return x


class TrajectoryEncoder(nn.Module):
    """
    Complete trajectory encoder with learnable queries

    Combines all trajectory attention components
    """

    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_action_queries: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_action_queries = num_action_queries

        # Action embedding
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # Learnable action queries
        self.action_queries = LearnableActionQueries(
            num_action_queries,
            hidden_dim,
            action_dim
        )

        # Trajectory transformer blocks
        self.blocks = nn.ModuleList([
            TrajectoryTransformerBlock(
                hidden_dim,
                num_heads,
                mlp_ratio,
                dropout,
                use_rope
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        context: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        num_actions: int = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Encode trajectories or generate new actions

        Args:
            context: Visual-language context [B, L, hidden_dim]
            actions: Optional previous actions [B, T, action_dim]
            num_actions: Number of actions to predict
            causal: Use causal masking

        Returns:
            Predicted actions [B, num_actions, action_dim]
        """
        # Get action queries
        if actions is not None:
            # Teacher forcing: use provided actions
            traj_embed = self.action_embed(actions)
        else:
            # Generation: use learnable queries
            if num_actions is None:
                num_actions = self.num_action_queries
            traj_embed, _ = self.action_queries(context, num_actions)

        # Apply trajectory transformer blocks
        for block in self.blocks:
            traj_embed = block(traj_embed, context, causal=causal)

        # Decode actions
        predicted_actions = self.output_proj(traj_embed)

        return predicted_actions

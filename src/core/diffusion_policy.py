"""
Diffusion Policy Module for VLA-GR
Based on latest SOTA research (Physical Intelligence π0, DP-VLA)
Enables stable, high-frequency action generation up to 50Hz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowMatchingBlock(nn.Module):
    """
    Flow Matching transformer block for continuous action generation
    Inspired by Physical Intelligence π0's flow matching approach
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Multi-head attention for flow matching
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Cross-attention with context if provided
        if context is not None:
            cross_attn_out, _ = self.attn(x, context, context)
            x = self.norm1(x + cross_attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class DiffusionPolicy(nn.Module):
    """
    Diffusion-based action policy for VLA-GR

    Key improvements:
    - Flow matching for continuous high-frequency actions (50Hz)
    - Conditional generation based on vision-language features
    - Fast sampling with DDIM (10-50 steps vs 1000 for DDPM)
    - Stable training with noise scheduling
    """

    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 256,
        context_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        num_diffusion_steps: int = 100,
        prediction_type: str = "v_prediction",  # v_prediction or epsilon
        dropout: float = 0.1,
        max_action_horizon: int = 16,  # Predict next 16 actions
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.prediction_type = prediction_type
        self.max_action_horizon = max_action_horizon

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Action embedding
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Context projection (from vision-language features)
        self.context_proj = nn.Linear(context_dim, hidden_dim)

        # Flow matching transformer blocks
        self.blocks = nn.ModuleList([
            FlowMatchingBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Noise schedule (cosine schedule)
        self.register_buffer(
            "betas",
            self._cosine_beta_schedule(num_diffusion_steps)
        )
        self.register_buffer(
            "alphas",
            1.0 - self.betas
        )
        self.register_buffer(
            "alphas_cumprod",
            torch.cumprod(self.alphas, dim=0)
        )

    def _cosine_beta_schedule(
        self,
        timesteps: int,
        s: float = 0.008
    ) -> torch.Tensor:
        """
        Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models"
        More stable than linear schedule
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward_diffusion(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0)

        Args:
            x_0: Clean actions [B, T, action_dim]
            t: Timesteps [B]
            noise: Optional noise [B, T, action_dim]

        Returns:
            x_t: Noised actions
            noise: The noise added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt()

        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise/velocity given noised actions and context

        Args:
            x_t: Noised actions [B, T, action_dim]
            t: Timesteps [B]
            context: Vision-language features [B, context_dim]

        Returns:
            prediction: Noise or velocity prediction [B, T, action_dim]
        """
        B, T, _ = x_t.shape

        # Embed timestep
        t_emb = self.time_mlp(t)  # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, hidden_dim]

        # Project action
        x_emb = self.action_proj(x_t)  # [B, T, hidden_dim]

        # Add time embedding
        x_emb = x_emb + t_emb

        # Project context
        context_emb = self.context_proj(context)  # [B, hidden_dim]
        context_emb = context_emb.unsqueeze(1)  # [B, 1, hidden_dim]

        # Apply flow matching blocks
        for block in self.blocks:
            x_emb = block(x_emb, context_emb)

        # Output projection
        prediction = self.output_proj(x_emb)

        return prediction

    def forward(
        self,
        actions: torch.Tensor,
        context: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass

        Args:
            actions: Ground truth actions [B, T, action_dim]
            context: Vision-language features [B, context_dim]

        Returns:
            Dictionary with loss and predictions
        """
        B, T, _ = actions.shape
        device = actions.device

        # Sample random timesteps
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=device)

        # Forward diffusion
        noise = torch.randn_like(actions)
        x_t, _ = self.forward_diffusion(actions, t, noise)

        # Predict noise or velocity
        prediction = self.predict_noise(x_t, t, context)

        if self.prediction_type == "epsilon":
            # Predict noise
            target = noise
        elif self.prediction_type == "v_prediction":
            # Predict velocity (more stable)
            sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(-1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1)
            target = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * actions
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # MSE loss
        loss = F.mse_loss(prediction, target)

        return {
            "loss": loss,
            "prediction": prediction,
            "target": target
        }

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        num_samples: int = 1,
        action_horizon: int = None,
        num_inference_steps: int = 10,
        eta: float = 0.0,  # DDIM parameter
    ) -> torch.Tensor:
        """
        Sample actions using DDIM for fast inference

        Args:
            context: Vision-language features [B, context_dim]
            num_samples: Number of action sequences to sample
            action_horizon: Number of future actions (default: max_action_horizon)
            num_inference_steps: Number of denoising steps (10-50 for speed)
            eta: DDIM stochasticity (0 = deterministic)

        Returns:
            Sampled actions [B, num_samples, T, action_dim]
        """
        if action_horizon is None:
            action_horizon = self.max_action_horizon

        B = context.shape[0]
        device = context.device

        # Start with random noise
        x_t = torch.randn(B, num_samples, action_horizon, self.action_dim, device=device)

        # Create inference timesteps (DDIM uses subset of training steps)
        step_size = self.num_diffusion_steps // num_inference_steps
        timesteps = torch.arange(0, self.num_diffusion_steps, step_size, device=device)
        timesteps = torch.flip(timesteps, [0])

        # Expand context for multiple samples
        context_expanded = context.unsqueeze(1).expand(-1, num_samples, -1)
        context_expanded = context_expanded.reshape(B * num_samples, -1)

        # Denoise iteratively
        for i, t in enumerate(timesteps):
            # Reshape for batch processing
            x_t_flat = x_t.reshape(B * num_samples, action_horizon, self.action_dim)
            t_batch = t.repeat(B * num_samples)

            # Predict noise
            noise_pred = self.predict_noise(x_t_flat, t_batch, context_expanded)

            # DDIM update
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                x_t_flat = self._ddim_step(
                    x_t_flat,
                    noise_pred,
                    t,
                    t_next,
                    eta
                )
            else:
                # Last step: predict x_0 directly
                alpha_t = self.alphas_cumprod[t]
                x_t_flat = (x_t_flat - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

            # Reshape back
            x_t = x_t_flat.reshape(B, num_samples, action_horizon, self.action_dim)

        return x_t

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        eta: float
    ) -> torch.Tensor:
        """DDIM sampling step"""
        alpha_t = self.alphas_cumprod[t]
        alpha_t_next = self.alphas_cumprod[t_next]

        # Predict x_0
        x_0_pred = (x_t - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

        # Direction pointing to x_t
        dir_xt = (1 - alpha_t_next - eta**2 * (1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t)).sqrt() * noise_pred

        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(x_t)
            sigma_t = eta * ((1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next)).sqrt()
            dir_xt = dir_xt + sigma_t * noise

        # Update
        x_t_next = alpha_t_next.sqrt() * x_0_pred + dir_xt

        return x_t_next

    @torch.no_grad()
    def get_action(
        self,
        context: torch.Tensor,
        num_inference_steps: int = 10
    ) -> torch.Tensor:
        """
        Get single action for execution (first action of sampled sequence)
        Fast inference optimized for real-time control

        Args:
            context: Vision-language features [B, context_dim]
            num_inference_steps: Number of denoising steps

        Returns:
            Action [B, action_dim]
        """
        # Sample action sequence
        action_seq = self.sample(
            context,
            num_samples=1,
            action_horizon=self.max_action_horizon,
            num_inference_steps=num_inference_steps,
            eta=0.0  # Deterministic for consistency
        )

        # Return first action
        return action_seq[:, 0, 0, :]  # [B, action_dim]

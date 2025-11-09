"""
Parameter-Efficient Fine-Tuning (PEFT) Modules for VLA-GR
Based on latest research: LoRA, OFT (Orthogonal Fine-Tuning)

Enables efficient fine-tuning with minimal trainable parameters:
- LoRA: 0.1-1% of parameters
- OFT: ~1% of parameters with better orthogonality preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer

    Decomposes weight updates into low-rank matrices:
    W' = W + BA where B is (d x r) and A is (r x k), r << min(d, k)

    This reduces trainable parameters from d*k to r*(d+k)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Scaling
        self.scaling = alpha / rank

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Adapted output [..., out_features]
        """
        # Apply dropout
        x_dropped = self.dropout(x)

        # Low-rank adaptation: x @ A^T @ B^T
        result = (x_dropped @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return result


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation

    During training: only LoRA parameters are updated
    During inference: can merge LoRA into base weights for no overhead
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        # Base linear layer (frozen during fine-tuning)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)

        # LoRA adaptation
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)

        # Flag for merged state
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # LoRA is merged into base weights
            return self.base_layer(x)
        else:
            # Separate computation
            return self.base_layer(x) + self.lora(x)

    def merge_lora(self):
        """Merge LoRA weights into base layer for inference"""
        if not self.merged:
            # Compute merged weight: W + B @ A * scaling
            with torch.no_grad():
                delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
                self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge_lora(self):
        """Unmerge LoRA weights from base layer"""
        if self.merged:
            with torch.no_grad():
                delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
                self.base_layer.weight.data -= delta_w
            self.merged = False

    def freeze_base(self):
        """Freeze base layer parameters"""
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        """Unfreeze base layer parameters"""
        for param in self.base_layer.parameters():
            param.requires_grad = True


class OFTLayer(nn.Module):
    """
    Orthogonal Fine-Tuning (OFT) layer

    Maintains orthogonality of weight matrices for better stability
    Based on "Controlling Text-to-Image Diffusion by Orthogonal Finetuning"

    Key idea: Constrain weight updates to lie on the orthogonal group
    W' = Q * W where Q is orthogonal (Q^T Q = I)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.eps = eps

        # Number of blocks
        assert out_features % rank == 0, "out_features must be divisible by rank"
        self.num_blocks = out_features // rank

        # Learnable skew-symmetric matrices (R in the paper)
        # We parameterize orthogonal matrices via Cayley transform
        self.oft_r = nn.Parameter(torch.zeros(self.num_blocks, rank, rank))

    def cayley_transform(self, R: torch.Tensor) -> torch.Tensor:
        """
        Cayley transform: (I - R)(I + R)^{-1}
        Maps skew-symmetric matrix to orthogonal matrix
        """
        # Make R skew-symmetric: R = R - R^T
        R_skew = R - R.transpose(-2, -1)

        I = torch.eye(R.size(-1), device=R.device, dtype=R.dtype)
        I = I.unsqueeze(0).expand_as(R)

        # Compute (I + R)^{-1}
        I_plus_R_inv = torch.linalg.inv(I + R_skew + self.eps * I)

        # Compute (I - R)(I + R)^{-1}
        Q = torch.matmul(I - R_skew, I_plus_R_inv)

        return Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply OFT transformation

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Transformed output [..., out_features]
        """
        batch_shape = x.shape[:-1]
        x = x.view(-1, self.in_features)

        # Get orthogonal transformation via Cayley transform
        Q = self.cayley_transform(self.oft_r)  # [num_blocks, rank, rank]

        # Create block-diagonal matrix from Q
        Q_block = torch.block_diag(*[Q[i] for i in range(self.num_blocks)])

        # Apply transformation
        result = x @ Q_block.T

        # Reshape back
        result = result.view(*batch_shape, self.out_features)

        return result


class OFTLinear(nn.Module):
    """
    Linear layer with OFT adaptation

    Recommended for VLA fine-tuning per latest research (2025)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        bias: bool = True,
    ):
        super().__init__()

        # Base linear layer (frozen during fine-tuning)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)

        # OFT adaptation
        self.oft = OFTLayer(in_features, out_features, rank)

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Base transformation
        base_out = self.base_layer(x)

        # OFT transformation (applied to base output)
        oft_out = self.oft(base_out)

        return oft_out

    def freeze_base(self):
        """Freeze base layer parameters"""
        for param in self.base_layer.parameters():
            param.requires_grad = False


class AdapterLayer(nn.Module):
    """
    Adapter layer for parameter-efficient fine-tuning
    Inserts small bottleneck modules into frozen networks

    Architecture: down_proj -> activation -> up_proj
    """

    def __init__(
        self,
        hidden_dim: int,
        adapter_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.down_proj = nn.Linear(hidden_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)

        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout)

        # Initialize with small values
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection

        Args:
            x: Input [B, ..., hidden_dim]

        Returns:
            Output [B, ..., hidden_dim]
        """
        # Adapter transformation
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)

        # Residual connection
        return residual + x


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "out_proj"],
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Replace target linear layers in model with LoRA versions

    Args:
        model: Model to modify
        target_modules: Names of modules to replace
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: LoRA dropout

    Returns:
        Modified model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this is a target module
            if any(target in name for target in target_modules):
                # Get parent module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model

                # Create LoRA replacement
                lora_layer = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    bias=module.bias is not None,
                )

                # Copy original weights
                lora_layer.base_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.base_layer.bias.data = module.bias.data.clone()

                # Freeze base layer
                lora_layer.freeze_base()

                # Replace module
                setattr(parent, child_name, lora_layer)

    return model


def apply_oft_to_model(
    model: nn.Module,
    target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "out_proj"],
    rank: int = 4,
) -> nn.Module:
    """
    Replace target linear layers in model with OFT versions

    Args:
        model: Model to modify
        target_modules: Names of modules to replace
        rank: OFT block rank

    Returns:
        Modified model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this is a target module
            if any(target in name for target in target_modules):
                # Get parent module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model

                # Create OFT replacement
                oft_layer = OFTLinear(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    bias=module.bias is not None,
                )

                # Copy original weights
                oft_layer.base_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    oft_layer.base_layer.bias.data = module.bias.data.clone()

                # Freeze base layer
                oft_layer.freeze_base()

                # Replace module
                setattr(parent, child_name, oft_layer)

    return model


def count_trainable_parameters(model: nn.Module) -> dict:
    """
    Count trainable vs total parameters

    Returns:
        Dictionary with parameter counts and percentages
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0,
    }

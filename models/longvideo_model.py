"""Core LongLive video generation model definition.

Builds on top of a diffusion transformer backbone with long-context
attention mechanisms for extended video synthesis.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any


class TemporalAttention(nn.Module):
    """Temporal self-attention with sliding window support for long sequences."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, heads, T, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class LongLiveBlock(nn.Module):
    """Transformer block combining spatial and temporal attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        window_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttention(
            dim, num_heads=num_heads, window_size=window_size, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.temporal_attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class LongLiveModel(nn.Module):
    """Long-context video generation model.

    Args:
        in_channels: Number of input latent channels.
        hidden_dim: Internal feature dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        window_size: Temporal attention window size.
        max_frames: Maximum number of frames supported.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        window_size: int = 32,
        max_frames: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames

        # Project latents into hidden dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        # Learnable temporal positional embedding
        self.temporal_pos_emb = nn.Embedding(max_frames, hidden_dim)

        self.blocks = nn.ModuleList([
            LongLiveBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                window_size=window_size,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, in_channels)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Latent tensor of shape (B, T, C).
            timestep: Diffusion timestep tensor (B,), unused placeholder for now.
            mask: Optional attention mask (B, T).

        Returns:
            Output latent tensor of shape (B, T, C).
        """
        B, T, _ = x.shape
        assert T <= self.max_frames, f"Input frames {T} exceed max_frames {self.max_frames}"

        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.input_proj(x) + self.temporal_pos_emb(pos_ids)

        for block in self.blocks:
            h = block(h, mask=mask)

        h = self.norm_out(h)
        return self.output_proj(h)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

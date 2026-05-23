"""
Context and target encoders  (§2.3, §E.3).

Both encoders use a point-transformer backbone with local self-attention.
The context encoder maps geometry point clouds → context tokens Zc.
The target encoder maps flow-field point clouds → target tokens Zt (training only).
"""

import torch
import torch.nn as nn
from .components import PointTransformerBlock, FourierFeatures
from ..data.tokenization import LearnedCentroidTokenization


class ContextEncoder(nn.Module):
    """
    Context encoder Ec — geometry point cloud → context tokens Zc.

    Reference (§2.3):
        "Given the subsampled geometry cloud P, the encoder produces a fixed
         number of context tokens, Zc = Ec(P) ∈ R^{M×d}"

    Architecture (§E.3):
        - Tokenisation via learned centroids (FPS + message-passing)
        - Point-transformer backbone (6 layers, local self-attention)
    """

    def __init__(
        self,
        in_channels: int = 3,          # point coordinates only (3) or + SDF (4)
        token_dim: int = 128,
        num_tokens: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_neighbours: int = 16,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens

        # Tokenisation layer (§E.2)
        self.tokenizer = LearnedCentroidTokenization(
            in_channels=in_channels,
            token_dim=token_dim,
            num_tokens=num_tokens,
            num_neighbours=num_neighbours,
        )

        # Point-transformer backbone (§E.3)
        self.blocks = nn.ModuleList([
            PointTransformerBlock(token_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(token_dim)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points:   (B, Nc, 3) subsampled geometry coordinates
            features: (B, Nc, C_in) point features (coords + optionally SDF)
        Returns:
            tokens:   (B, M, d) context latent tokens
        """
        x = self.tokenizer(points, features)  # (B, M, d)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x  # (B, M, d)


class TargetEncoder(nn.Module):
    """
    Target encoder Et — flow-field point cloud → target tokens Zt.

    Reference (§2.3):
        "The target encoder maps the subsampled flow field F to target tokens,
         Zt = Et(F) ∈ R^{M×d}"

    Used only during training.  Architecture mirrors the context encoder.
    """

    def __init__(
        self,
        in_channels: int,               # typically 3 + num_flow_vars (e.g., 3+4=7 for u,v,w,p)
        token_dim: int = 128,
        num_tokens: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_neighbours: int = 16,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens

        self.tokenizer = LearnedCentroidTokenization(
            in_channels=in_channels,
            token_dim=token_dim,
            num_tokens=num_tokens,
            num_neighbours=num_neighbours,
        )

        self.blocks = nn.ModuleList([
            PointTransformerBlock(token_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(token_dim)

    def forward(self, points: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, Nt, 3) subsampled flow coordinates
            fields: (B, Nt, C_flow) flow quantities (coords + u,v,w,p or Cp,Cf...)
        Returns:
            tokens: (B, M, d) target latent tokens
        """
        x = self.tokenizer(points, fields)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

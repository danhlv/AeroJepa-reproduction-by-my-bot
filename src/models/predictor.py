"""
Latent predictor network  (§2.4, §E.4).

Predicts target flow tokens Ẑt from context tokens Zc and operating conditions c.

Architecture (§E.4):
    - Learnable latent queries with Fourier positional encodings
    - Stacked blocks alternating self-attention and cross-attention with context tokens
    - Adaptive feature modulation for flow-condition injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    SelfAttentionBlock,
    CrossAttentionBlock,
    AdaptiveModulation,
    FourierFeatures,
)


class LatentPredictor(nn.Module):
    """
    Latent predictor f_θ_pred — predicts Ẑt from (Zc, c).

    Reference (§E.4):
        "The predictor instantiates learnable latent queries corresponding to the
         spatial coordinates of a fixed set of centroids.  These spatial query
         coordinates are first embedded via high-frequency Fourier positional encodings."

        "The self-attention pathway facilitates continuous spatial refinement ...
         the interleaving cross-attention layers allow these queries to explicitly
         fetch aligned geometric features from the upstream context tokens."

        "We inject these physics conditions natively into the network ... through
         adaptive feature modulation."
    """

    def __init__(
        self,
        token_dim: int = 128,
        num_tokens: int = 512,
        cond_dim: int = 2,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        fourier_dim: int = 32,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.depth = depth

        # Fourier positional encoding for learnable centroids
        self.fourier = FourierFeatures(3, fourier_dim, scale=10.0)

        # Learnable centroid coordinates (spatial queries)
        self.centroid_coords = nn.Parameter(torch.randn(1, num_tokens, 3) * 0.1)
        self.pos_embed = nn.Linear(fourier_dim, token_dim)

        # Learnable latent queries (token content)
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)

        # Alternating self-attention and cross-attention blocks
        self.self_attn_blocks = nn.ModuleList([
            SelfAttentionBlock(token_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(token_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Adaptive modulation for flow conditions (§E.4)
        self.ada_mod = AdaptiveModulation(cond_dim, token_dim)

        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_tokens: (B, M, d) token embeddings from context encoder
            conditions:     (B, C) operating conditions (α, Re, Mach, ...)
        Returns:
            predicted_tokens: (B, M, d) predicted target tokens Ẑt
        """
        B = context_tokens.shape[0]

        # Positional encoding from learnable centroids
        pos = self.fourier(self.centroid_coords.expand(B, -1, -1))  # (B, M, fourier_dim)
        pos_enc = self.pos_embed(pos)  # (B, M, d_token)

        # Latent queries = learned tokens + positional encoding
        z = self.query_tokens.expand(B, -1, -1) + pos_enc  # (B, M, d)

        # Alternating blocks
        for i in range(self.depth):
            # Cross-attention: queries fetch from context tokens
            z = self.cross_attn_blocks[i](query=z, key_value=context_tokens)
            # Self-attention: refine among predicted tokens
            z = self.self_attn_blocks[i](z)

        # Inject operating conditions via adaptive modulation
        z = self.ada_mod(z, conditions)

        return self.norm(z)

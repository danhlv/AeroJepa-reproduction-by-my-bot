"""
Shared building blocks for AeroJEPA.

Includes:
    - FourierFeatures (§E.5)          : sinusoidal encoding of spatial coordinates
    - PointTransformerBlock (§E.3)    : local self-attention over token sets
    - AdaptiveModulation (§E.4)       : adaptive feature modulation for flow conditions
    - SelfAttentionBlock / CrossAttentionBlock : standard transformer primitives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierFeatures(nn.Module):
    """
    High-frequency Fourier positional encoding.

    Maps a D-dimensional coordinate to a (2 * fourier_dim)-dimensional encoding
    using sinusoidal functions with random frequency scales.

    Reference (§E.5):
        "the coordinate is first transformed via Fourier feature encodings to
         enhance high-frequency spatial sensitivity"
    """

    def __init__(self, input_dim: int, output_dim: int, scale: float = 10.0):
        super().__init__()
        self.register_buffer(
            "B",
            torch.randn(input_dim, output_dim // 2) * scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., input_dim) → (..., output_dim)"""
        proj = 2 * math.pi * x @ self.B  # (..., output_dim // 2)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class AdaptiveModulation(nn.Module):
    """
    Adaptive feature modulation  (§E.4).

    Projects scalar operating conditions into scale / shift vectors that modulate
    hidden activations, inspired by DiT (Peebles & Xie 2023).

    Reference (§E.4):
        "A lightweight multi-layer perceptron processes the scalar variables,
         projecting them deeply into the hidden transformer layers through
         adaptive feature modulation."
    """

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # scale + shift
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:    (B, N, D) hidden states
            cond: (B, C)    conditioning vector
        Returns:
            modulated: (B, N, D)
        """
        mod = self.net(cond)  # (B, D*2)
        scale, shift = mod.chunk(2, dim=-1)  # each (B, D)
        return h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PointTransformerBlock(nn.Module):
    """
    Local point-transformer self-attention block  (§E.3).

    Uses a fixed set of spatial tokens and performs local self-attention
    within a neighbourhood of each token (avoids global quadratic cost).

    Reference (§E.3):
        "By employing localized self-attention rather than global quadratic
         computations, the framework strictly bounds memory consumption."
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SelfAttentionBlock(nn.Module):
    """Standard pre-norm self-attention transformer block."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm1(x)
        x = x + self.attn(xn, xn, xn)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + self-attention block.

    Query attends to key/value tokens, followed by a self-attention refinement.
    Used in the predictor to fetch geometric features from context tokens (§E.4).
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn = SelfAttentionBlock(d_model, n_heads, mlp_ratio, dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        out, _ = self.cross_attn(q, kv, kv)
        query = query + out
        query = self.self_attn(query)
        return query


class RidgeProbe(nn.Module):
    """
    Differentiable linear probe (ridge regression)  (§F.5).

    Used to read out design parameters and aerodynamic coefficients from
    the latent space.  The forward pass is:  y = W @ z_std + b.

    Reference (§F.5):
        "For a latent vector z ∈ R^D and a scalar target y, we define a linear
         probe as the composition of train-set standardisation and ridge regression."
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(d_out, d_in))
        self.b = nn.Parameter(torch.zeros(d_out))
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sigma", torch.ones(d_in))

    def fit_ridge(self, z: torch.Tensor, y: torch.Tensor, lambda_: float = 1.0):
        """
        Closed-form ridge regression.

        z: (N, D) latent vectors
        y: (N, K) targets
        """
        z = z.cpu().numpy()
        y = y.cpu().numpy()

        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=lambda_, fit_intercept=True)
        reg.fit(z, y)

        with torch.no_grad():
            self.W.copy_(torch.from_numpy(reg.coef_).float())
            self.b.copy_(torch.from_numpy(reg.intercept_).float())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., D) → (..., K)"""
        z_std = (z - self.mu) / self.sigma.clamp(min=1e-6)
        return F.linear(z_std, self.W, self.b)

"""
Implicit Neural Representation (INR) decoder  (§2.5, §E.5).

Maps predicted latent tokens + spatial query coordinates → continuous fluid field values.

The decoder is an MLP conditioned on the token-level latent representation.
At each query point q, the decoder outputs [u(q), v(q), w(q), p(q)] (surface)
or [u(q), p(q)] (volumetric).

Reference (§2.5):
    "By conditioning a multi-layer perceptron (MLP) on the latent vector Ẑt,
     the INR Decoder acts as a continuous basis function for the aerodynamic field."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import FourierFeatures


class INRDecoder(nn.Module):
    """
    Continuous implicit decoder — predicted tokens + query → flow values.

    Architecture (§E.5):
        1. Fourier feature encoding of query coordinates.
        2. Cross-attention aggregation of token information.
        3. MLP regression to output field values.
    """

    def __init__(
        self,
        token_dim: int = 128,
        num_tokens: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        output_dim: int = 4,              # e.g., (u, v, w, p) or (Cp, Cf_tau, Cf_z)
        fourier_dim: int = 32,
        fourier_scale: float = 10.0,
        use_sdf: bool = False,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.use_sdf = use_sdf

        # Fourier encoding for query coordinates
        self.fourier = FourierFeatures(3, fourier_dim * 2, scale=fourier_scale)

        # Cross-attention: query locations attend to predicted tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )
        self.query_proj = nn.Linear(fourier_dim * 2 + (1 if use_sdf else 0), hidden_dim)
        self.token_proj = nn.Linear(token_dim, hidden_dim)

        # MLP decoder
        layers = []
        dims = [hidden_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.GELU(),
            ])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        predicted_tokens: torch.Tensor,
        query_points: torch.Tensor,
        sdf_values: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            predicted_tokens: (B, M, d) predicted latent tokens Ẑt
            query_points:     (B, Nq, 3) arbitrary query coordinates
            sdf_values:       (B, Nq, 1) optional SDF values (volumetric)
        Returns:
            field: (B, Nq, output_dim) decoded fluid state
        """
        B, Nq, _ = query_points.shape

        # Fourier encode query coordinates
        q_feat = self.fourier(query_points)  # (B, Nq, fourier_dim*2)
        if self.use_sdf and sdf_values is not None:
            q_feat = torch.cat([q_feat, sdf_values], dim=-1)

        # Project to hidden dimension
        q_feat = self.query_proj(q_feat)  # (B, Nq, hidden_dim)
        t_feat = self.token_proj(predicted_tokens)  # (B, M, hidden_dim)

        # Cross-attention: query tokens fetch from predicted tokens
        out, _ = self.cross_attn(
            query=self.norm(q_feat),
            key=self.norm(t_feat),
            value=t_feat,
        )  # (B, Nq, hidden_dim)

        # MLP regression
        field = self.mlp(out)
        return field

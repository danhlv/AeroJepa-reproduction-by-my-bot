"""
Tokenization by learned centroids  (§E.2).

Converts irregular subsampled point clouds into fixed-length sequences of latent tokens
via learned centroid clustering and local message-passing neighbourhood aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocessing import farthest_point_sampling


class LearnedCentroidTokenization(nn.Module):
    """
    Turns a variable-size point cloud into fixed M tokens.

    Pipeline:
        1. FPS over the point cloud to select M*scale centroid locations.
        2. For each centroid, gather K nearest neighbours.
        3. Lightweight message-passing network aggregates neighbour features.
        4. A point-transformer backbone refines the M token set.

    Reference (§E.2):
        "The architecture leverages FPS once more over the subsampled sets to designate
         foundational centroid coordinates. A localized message-passing neighborhood
         aggregation layer compiles point-wise variables around each centroid..."
    """

    def __init__(
        self,
        in_channels: int,
        token_dim: int,
        num_tokens: int,
        num_neighbours: int = 16,
        fps_scale: float = 2.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_neighbours = num_neighbours
        self.num_centroids = int(num_tokens * fps_scale)

        # Project each point's features before aggregation
        self.point_proj = nn.Linear(in_channels, token_dim)

        # Message-passing aggregation MLP
        self.aggr_mlp = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim),
        )

        # Cross-attention: centroids query the aggregated tokens → produce M tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=4,
            batch_first=True,
        )
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)

        # Layer norm
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points:   (B, N, 3) spatial coordinates.
            features: (B, N, C_in) point features (coords + optional SDF).
        Returns:
            tokens:   (B, M, d_token) token embeddings.
        """
        B, N, _ = points.shape

        # 1. FPS for centroid locations
        centroids, _ = farthest_point_sampling(points, self.num_centroids)  # (B, n_cent, 3)

        # 2. Find K nearest neighbours for each centroid
        #    (B, n_cent, N) distance matrix via efficient pairwise
        diff = centroids.unsqueeze(2) - points.unsqueeze(1)  # (B, n_cent, N, 3)
        dists = diff.norm(dim=-1)  # (B, n_cent, N)
        _, nn_idx = dists.topk(k=self.num_neighbours, dim=-1, largest=False)  # (B, n_cent, K)

        # 3. Gather neighbour features and aggregate
        #    nn_idx: (B, n_cent, K) → gather from features: (B, N, C_in)
        #    Expand indices for gather
        idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, -1, features.size(-1))
        neigh_feats = features.unsqueeze(1).expand(-1, self.num_centroids, -1, -1)
        neigh_feats = torch.gather(neigh_feats, 2, idx_exp)  # (B, n_cent, K, C_in)

        # Project point features
        neigh_feats = self.point_proj(neigh_feats)  # (B, n_cent, K, d_token)

        # Centroid feature: mean of neighbours
        centroid_feats = neigh_feats.mean(dim=2)  # (B, n_cent, d_token)

        # Message-passing: concat centroid with mean neighbour & MLP
        msg = torch.cat([centroid_feats, neigh_feats.mean(dim=2)], dim=-1)
        agg_tokens = self.aggr_mlp(msg)  # (B, n_cent, d_token)

        # 4. Cross-attention: fixed query tokens attend to aggregated tokens
        q = self.query_tokens.expand(B, -1, -1)  # (B, M, d_token)
        out, _ = self.cross_attn(
            query=q,
            key=self.norm(agg_tokens),
            value=agg_tokens,
        )

        return out  # (B, M, d_token)

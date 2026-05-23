"""
Point-cloud preprocessing and subsampling.

Implements the Farthest Point Sampling (FPS) and SDF computation used to transform
raw CFD meshes into the decoupled point-cloud representation described in §E.1.
"""

import torch
import numpy as np
from typing import Optional


def farthest_point_sampling(
    points: torch.Tensor,
    n_samples: int,
    start_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) over a batch of point clouds.

    Args:
        points: (B, N, D) tensor of coordinates / features.
        n_samples: Number of points to sample.
        start_idx: Optional (B,) tensor of starting indices, or None (random).

    Returns:
        sampled: (B, n_samples, D) tensor.
        indices: (B, n_samples) long tensor of chosen indices.

    Reference:
        §E.1 — "Utilizing a Farthest Point Sampling (FPS) heuristic to ensure
        globally comprehensive spatial coverage."
    """
    B, N, D = points.shape
    device = points.device

    if n_samples >= N:
        # Pad with repeated points if the cloud is already smaller
        idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        if n_samples > N:
            repeat = n_samples - N
            extra = idx[:, :repeat]
            idx = torch.cat([idx, extra], dim=-1)
        return points.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)), idx

    # Distance buffer (B, N)
    dists = torch.full((B, N), float("inf"), device=device)
    selected = torch.zeros((B, N), dtype=torch.bool, device=device)

    if start_idx is not None:
        curr = start_idx  # (B,)
    else:
        curr = torch.randint(0, N, (B,), device=device)

    indices = [curr.clone()]

    for _ in range(1, n_samples):
        # Compute distance from current point to all points
        curr_pts = points.gather(1, curr.view(B, 1, 1).expand(-1, -1, D)).squeeze(1)  # (B, D)
        delta = points - curr_pts.unsqueeze(1)  # (B, N, D)
        d = delta.norm(dim=-1)  # (B, N)
        dists = torch.min(dists, d)
        dists[selected] = -1.0  # mask already selected

        # Pick farthest
        curr = dists.argmax(dim=-1)  # (B,)
        selected.scatter_(1, curr.unsqueeze(1), True)
        indices.append(curr.clone())

    idx = torch.stack(indices, dim=-1)  # (B, n_samples)
    return points.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)), idx


def compute_sdf_batch(
    surface_points: torch.Tensor,
    query_points: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate signed-distance function via nearest-neighbour on the surface.

    Args:
        surface_points: (B, N_surf, 3) surface point cloud.
        query_points:   (B, N_q, 3) query coordinates.

    Returns:
        sdf: (B, N_q) signed-distance values (positive = outside).
             Sign is determined by dot product with approximate surface normal.

    Note:
        The paper uses exact SDF from the CFD mesh.  This is an approximation
        for standalone use; replace with mesh-based SDF for production.
    """
    B, N_q = query_points.shape[:2]
    dists, idx = pairwise_dist(surface_points, query_points)  # (B, N_q)
    closest_pts = surface_points.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))

    # Approximate outward normal as vector from centroid to closest point
    centroid = surface_points.mean(dim=1, keepdim=True)  # (B, 1, 3)
    normals = (closest_pts - centroid)  # (B, N_q, 3)
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)

    delta = query_points - closest_pts
    sign = (delta * normals).sum(dim=-1).sign()
    return sign * torch.sqrt(dists + 1e-8)


def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> tuple:
    """Efficient squared pairwise distance between two point sets."""
    xx = (x ** 2).sum(dim=-1, keepdim=True)   # (B, Nx, 1)
    yy = (y ** 2).sum(dim=-1, keepdim=True).transpose(-2, -1)  # (B, 1, Ny)
    xy = torch.bmm(x, y.transpose(-2, -1))    # (B, Nx, Ny)
    dists = xx - 2 * xy + yy                   # (B, Nx, Ny)
    d, idx = dists.min(dim=-2)                 # (B, Ny)
    return d, idx


def normalize_point_cloud(
    points: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> tuple:
    """
    Normalise point coordinates to zero mean and unit variance.

    Args:
        points: (B, N, D) tensor.
        mean, std: optional pre-computed statistics.

    Returns:
        points_norm, mean, std
    """
    if mean is None:
        mean = points.mean(dim=(0, 1), keepdim=True)
    if std is None:
        std = points.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    return (points - mean) / std, mean, std

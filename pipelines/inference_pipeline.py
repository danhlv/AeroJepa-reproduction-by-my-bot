#!/usr/bin/env python3
"""
Inference pipeline  (§2.2).

Given a trained AeroJEPA model, reconstruct flow fields at arbitrary query locations.

Inference:
    1. Encode geometry point cloud → context tokens (once)
    2. Predict target tokens from context + operating conditions
    3. Decode field at arbitrary query points via INR decoder

Usage:
    python pipelines/inference_pipeline.py --checkpoint /path/to/best.pt \
        --geometry data/superwing/test/geometry_001.npz \
        --conditions "0.0" --output field_prediction.vtk

Reference (§2.2):
    "At inference time, target encoder is discarded."
"""

import argparse
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AeroJEPAConfig
from src.models.aerojepa import AeroJEPA
from src.data.preprocessing import farthest_point_sampling, normalize_point_cloud


def main():
    parser = argparse.ArgumentParser(description="AeroJEPA Inference Pipeline")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--geometry", type=str, required=True,
                        help="Path to geometry .npz file (contains 'surface_pts' and optionally 'sdf')")
    parser.add_argument("--conditions", type=float, nargs="+", required=True,
                        help="Operating conditions (e.g., --conditions 4.0 0.3 for AoA=4°, Mach=0.3)")
    parser.add_argument("--query", type=str, default=None,
                        help="Path to query points .npz (optional; if omitted, generates a coarse grid)")
    parser.add_argument("--output", type=str, default="./field_prediction.npz",
                        help="Output path for predicted field (.npz)")
    parser.add_argument("--nc", type=int, default=8192,
                        help="Context subsample count")
    parser.add_argument("--nq", type=int, default=32000,
                        help="Number of query points (if generating grid)")
    parser.add_argument("--dataset", type=str, default="superwing",
                        choices=["superwing", "hiliftaeroml"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load config and model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", AeroJEPAConfig(dataset=args.dataset))

    model = AeroJEPA(
        context_in_channels=3,
        target_in_channels=6 if "superwing" in args.dataset else 7,
        token_dim=cfg.d_token,
        num_tokens=cfg.m_tokens,
        encoder_depth=cfg.encoder_depth,
        encoder_heads=cfg.encoder_heads,
        cond_dim=cfg.condition_dim,
        predictor_depth=cfg.predictor_depth,
        predictor_heads=cfg.predictor_heads,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        decoder_num_layers=cfg.decoder_num_layers,
        decoder_output_dim=3 if "superwing" in args.dataset else 4,
        fourier_dim=cfg.fourier_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Load geometry
    data = np.load(args.geometry)
    pts = torch.from_numpy(data["surface_pts"]).float().unsqueeze(0).to(device)  # (1, N, 3)
    sdf = torch.from_numpy(data.get("sdf", np.zeros((pts.shape[1], 1)))).float().unsqueeze(0).to(device)

    # Subsample geometry
    pts_sampled, _ = farthest_point_sampling(pts, args.nc)
    pts_sampled, mean, std = normalize_point_cloud(pts_sampled)

    # Features: coordinates + optional SDF
    feat_sampled = torch.cat([pts_sampled, sdf[:, :args.nc]], dim=-1) if sdf is not None else pts_sampled

    # Conditions
    conditions = torch.tensor([args.conditions], device=device)

    # Prepare query points
    if args.query:
        qdata = np.load(args.query)
        query_pts = torch.from_numpy(qdata["query_pts"]).float().unsqueeze(0).to(device)
        # Normalise with same stats as geometry
        query_pts = (query_pts - mean) / std.clamp(min=1e-6)
    else:
        # Generate a coarse grid around the geometry
        bb_min = pts_sampled.min(dim=1)[0].squeeze().cpu().numpy()
        bb_max = pts_sampled.max(dim=1)[0].squeeze().cpu().numpy()
        n = int(round(args.nq ** (1/3)))  # cube root
        x = np.linspace(bb_min[0] - 0.5, bb_max[0] + 0.5, n)
        y = np.linspace(bb_min[1] - 0.5, bb_max[1] + 0.5, n)
        z = np.linspace(bb_min[2] - 0.5, bb_max[2] + 0.5, n)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        query_pts = torch.from_numpy(grid).float().unsqueeze(0).to(device)

    print(f"Geometry: {pts_sampled.shape}")
    print(f"Query points: {query_pts.shape}")
    print(f"Conditions: {conditions.tolist()}")

    # Inference
    with torch.no_grad():
        out = model(
            geometry_points=pts_sampled,
            geometry_features=feat_sampled,
            conditions=conditions,
            query_points=query_pts,
        )

    field = out["field"].cpu().numpy().squeeze()  # (Nq, C)

    # Save results
    np.savez(
        args.output,
        query_pts=query_pts.cpu().numpy().squeeze(),
        predicted_field=field,
        conditions=np.array(args.conditions),
    )
    print(f"Predicted field saved to {args.output}")
    print(f"Field shape: {field.shape}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Latent-space analysis pipeline  (§3.2, §F.1–F.5).

Analyses a trained AeroJEPA model from three perspectives:
    1. Linear probing: recover design parameters and aerodynamic coefficients
    2. Concept-vector arithmetic: disentanglement analysis
    3. Latent interpolation: smooth transitions between operating conditions

Usage:
    python pipelines/latent_analysis_pipeline.py \
        --checkpoint /path/to/best.pt \
        --dataset superwing \
        --data-root ./data \
        --output ./analysis_results

Reference:
    §F.5: "The probes are the bridge between the self-supervised representation
           and the design-space quantities a domain expert cares about."
"""

import argparse
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AeroJEPAConfig
from src.models.aerojepa import AeroJEPA
from src.data.datasets import HiLiftAeroMLDataset, SuperWingDataset, collate_aerojepa
from src.analysis.probing import fit_ridge_probes, cv_ridge_r2, concept_vectors
from src.analysis.latent_analysis import LatentAnalyzer
from src.analysis.visualization import (
    plot_latent_pca,
    plot_probe_recovery,
    plot_concept_matrix,
)


def main():
    parser = argparse.ArgumentParser(description="AeroJEPA Latent Analysis Pipeline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="superwing",
                        choices=["superwing", "hiliftaeroml"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./analysis_results")
    parser.add_argument("--r2-threshold", type=float, default=0.85,
                        help="R² threshold for high-quality design probes")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", AeroJEPAConfig(dataset=args.dataset))

    if "hilift" in args.dataset.lower():
        decoder_out = 4
        target_in = 7
    else:
        decoder_out = 3
        target_in = 6

    model = AeroJEPA(
        context_in_channels=3,
        target_in_channels=target_in,
        token_dim=cfg.d_token,
        num_tokens=cfg.m_tokens,
        encoder_depth=cfg.encoder_depth,
        encoder_heads=cfg.encoder_heads,
        cond_dim=cfg.condition_dim,
        predictor_depth=cfg.predictor_depth,
        predictor_heads=cfg.predictor_heads,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        decoder_num_layers=cfg.decoder_num_layers,
        decoder_output_dim=decoder_out,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Data
    Dataset = HiLiftAeroMLDataset if "hilift" in args.dataset.lower() else SuperWingDataset
    dataset = Dataset(root=args.data_root, split="train", nc=cfg.nc, nt=cfg.nt, nq=cfg.nq)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_aerojepa)

    # 1. Extract latents
    print("Extracting latents...")
    analyzer = LatentAnalyzer(model, device=device)
    latents = analyzer.extract_latents(loader, use_predicted=True)

    z_ctx = latents["ctx"]
    z_pred = latents["pred"]
    design = latents["design"]  # (N, K)
    cond = latents["conditions"]

    print(f"Context latents: {z_ctx.shape}")
    print(f"Predicted latents: {z_pred.shape}")
    print(f"Design params: {design.shape}")

    # 2. PCA projection
    proj, pca = analyzer.pca_projection(z_ctx)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    plot_latent_pca(
        proj,
        cond[:, 0] if cond.shape[1] > 0 else np.zeros(proj.shape[0]),
        title=f"Context Latent PCA — {args.dataset.upper()}",
        save_path=os.path.join(args.output, "pca_context.png"),
        colorbar_label="AoA" if "hilift" in args.dataset.lower() else "Condition",
    )

    # 3. Linear probes: context → design params
    print("Fitting context → design probes...")
    W_design, b_design, r2_design = fit_ridge_probes(z_ctx, design)

    print(f"  Design probe R²: min={r2_design.min():.4f}, "
          f"max={r2_design.max():.4f}, mean={r2_design.mean():.4f}")
    high_quality = (r2_design >= args.r2_threshold).sum()
    print(f"  High-quality probes (R² ≥ {args.r2_threshold}): {high_quality}/{len(r2_design)}")

    plot_probe_recovery(
        design, z_ctx @ W_design.T + b_design,
        title=f"Context → Design Probe Recovery — {args.dataset.upper()}",
        save_path=os.path.join(args.output, "probe_design.png"),
    )

    # 4. Linear probes: predicted → aerodynamic coefficients
    #    (For SuperWing we approximate CL/CD; for HiLift we use the flow state info)
    print("\nComputing CV R² for predicted → aerodynamic probes...")
    cv_r2 = cv_ridge_r2(z_pred, design[:, :2])  # first two design dims as proxy
    print(f"  Predicted latent — target R² (CV): {cv_r2}")

    # 5. Concept-vector arithmetic
    print("\nComputing concept-vector disentanglement...")
    param_names = [f"Param_{i}" for i in range(design.shape[1])]

    concepts = concept_vectors(W_design)
    disent = analyzer.concept_arithmetic(
        z_ctx, W_design, design, param_names[:4],  # first 4 params
    )
    plot_concept_matrix(
        disent["matrix"],
        disent["param_names"][:4],
        title=f"Concept-Vector Disentanglement — {args.dataset.upper()}",
        save_path=os.path.join(args.output, "concept_matrix.png"),
    )

    # 6. Latent interpolation (between two random test samples)
    print("\nLatent interpolation example...")
    z_a = z_ctx[0]
    z_b = z_ctx[min(10, z_ctx.shape[0] - 1)]
    interp = analyzer.interpolate_latents(z_a, z_b, n_steps=10)
    print(f"  Interpolation path: {interp.shape}")

    # Save all analysis results
    np.savez(
        os.path.join(args.output, "analysis_results.npz"),
        z_ctx=z_ctx,
        z_pred=z_pred,
        design=design,
        W_design=W_design,
        b_design=b_design,
        r2_design=r2_design,
        pca_explained=pca.explained_variance_ratio_,
        pca_components=pca.components_,
    )
    print(f"\nAll results saved to {args.output}/")


if __name__ == "__main__":
    main()

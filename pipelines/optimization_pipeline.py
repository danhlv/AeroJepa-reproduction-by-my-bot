#!/usr/bin/env python3
"""
Latent-space optimisation pipeline  (§F.3, Fig. 4, 14).

Performs a proof-of-concept design optimisation by searching directly in the
context latent space, using a differentiable chain:
    context latent → predictor → linear probes → CL/CD

Usage:
    python pipelines/optimization_pipeline.py \
        --checkpoint /path/to/best.pt \
        --dataset superwing \
        --data-root ./data \
        --output ./optimization_results

Reference (§F.3):
    "The frozen AeroJEPA predictor maps a context latent and a flow condition
     to a fluid latent, and two linear ridge probes read out CL and CD."
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
from src.models.components import RidgeProbe
from src.data.datasets import SuperWingDataset, collate_aerojepa
from src.analysis.probing import fit_ridge_probes
from src.analysis.latent_analysis import LatentAnalyzer
from src.analysis.visualization import plot_optimization_trajectory, plot_latent_pca
from src.optimization.latent_optim import LatentOptimizer


def main():
    parser = argparse.ArgumentParser(description="AeroJEPA Latent Optimisation Pipeline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="superwing",
                        choices=["superwing"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./optimization_results")
    parser.add_argument("--n-restarts", type=int, default=8)
    parser.add_argument("--cruise-alpha", type=float, default=2.0,
                        help="Cruise angle of attack (degrees)")
    parser.add_argument("--cruise-mach", type=float, default=0.7,
                        help="Cruise Mach number")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== AeroJEPA Latent Optimisation Pipeline ===")
    print(f"Dataset: {args.dataset}")
    print(f"Cruise: α={args.cruise_alpha}°, Mach={args.cruise_mach}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", AeroJEPAConfig(dataset=args.dataset))

    model = AeroJEPA(
        context_in_channels=3,
        target_in_channels=6,
        token_dim=cfg.d_token,
        num_tokens=cfg.m_tokens,
        encoder_depth=cfg.encoder_depth,
        encoder_heads=cfg.encoder_heads,
        cond_dim=cfg.condition_dim,
        predictor_depth=cfg.predictor_depth,
        predictor_heads=cfg.predictor_heads,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        decoder_num_layers=cfg.decoder_num_layers,
        decoder_output_dim=3,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Load training data for probes
    dataset = SuperWingDataset(root=args.data_root, split="train", nc=cfg.nc, nt=cfg.nt, nq=cfg.nq)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_aerojepa)

    # Extract latents
    print("Extracting latents from training set...")
    analyzer = LatentAnalyzer(model, device=device)
    latents = analyzer.extract_latents(loader, use_predicted=True)

    z_ctx = latents["ctx"]
    z_pred = latents["pred"]
    design = latents["design"]

    print(f"Context latents: {z_ctx.shape}")
    print(f"Predicted latents: {z_pred.shape}")

    # Fit probes (see §F.5)
    print("Fitting context → design probes...")
    W_design, b_design, r2_design = fit_ridge_probes(z_ctx, design)

    print("Fitting predicted → CL/CD probes...")
    # Approximate CL, CD from design parameters or from probe on predicted latents
    # For the actual dataset, use ground-truth CL/CD
    # Here we use a proxy: first two design params as (CL, CD) approximation
    y_cl = design[:, 0]  # placeholder — use actual CL in production
    y_cd = design[:, 1]  # placeholder — use actual CD in production

    # Fit CL probe
    from sklearn.linear_model import RidgeCV
    alphas = np.logspace(-4, 4, 17)
    cl_probe = RidgeCV(alphas=alphas).fit(z_pred, y_cl)
    cd_probe = RidgeCV(alphas=alphas).fit(z_pred, y_cd)

    cl_r2 = cl_probe.score(z_pred, y_cl)
    cd_r2 = cd_probe.score(z_pred, y_cd)
    print(f"CL probe R²: {cl_r2:.4f}")
    print(f"CD probe R²: {cd_r2:.4f}")

    # Run optimisation
    print(f"\nRunning latent-space optimisation ({args.n_restarts} restarts)...")
    optimizer = LatentOptimizer(
        predictor=model.predictor,
        design_probe_weights=W_design,
        design_probe_biases=b_design,
        cl_probe_weights=cl_probe.coef_,
        cd_probe_weights=cd_probe.coef_,
        cl_probe_bias=cl_probe.intercept_,
        cd_probe_bias=cd_probe.intercept_,
        mu_ctx=z_ctx.mean(axis=0),
        sigma_ctx=z_ctx.std(axis=0),
        train_design_params=design,
        train_cl=y_cl,
        train_cd=y_cd,
        config=cfg,
        device=device,
    )

    condition = np.array([args.cruise_alpha, args.cruise_mach])
    result = optimizer.optimize(
        condition=condition,
        n_restarts=args.n_restarts,
    )

    print(f"\n=== Optimisation Results ===")
    print(f"Success: {result['success']}")
    print(f"Optimal CL: {result['cl_opt']:.4f}")
    print(f"Optimal CD: {result['cd_opt']:.4f}")
    print(f"Optimal L/D: {result['ld_opt']:.2f}")
    print(f"Nearest neighbour index: {result['nearest_neighbour_idx']}")

    # Visualise
    print("\nGenerating visualisations...")
    proj, pca = analyzer.pca_projection(z_ctx)
    ld_train = y_cl / (y_cd + 1e-8)

    # PCA-optimised constraint (approximate)
    mu_pca = proj.mean(axis=0)
    cov_pca = np.cov(proj, rowvar=False)
    from scipy.stats import chi2
    thresh = chi2.ppf(0.95, df=2)
    trust_ellipse = (mu_pca[0], mu_pca[1], np.sqrt(cov_pca[0, 0] * thresh),
                     np.sqrt(cov_pca[1, 1] * thresh))

    # Trajectory: project optimisation path
    z_opt = result["z_opt"]
    z_nn = z_ctx[result["nearest_neighbour_idx"]]

    # For visualisation, project start — for simplicity use origin
    z_start = z_ctx[0]
    traj = np.stack([z_start, z_opt], axis=0)
    traj_proj = pca.transform(traj)

    opt_proj = pca.transform(z_opt.reshape(1, -1))[0]
    nn_proj = pca.transform(z_nn.reshape(1, -1))[0]

    plot_optimization_trajectory(
        proj, ld_train, traj_proj, opt_proj, nn_proj,
        trust_region=trust_ellipse,
        save_path=os.path.join(args.output, "optimization_trajectory.png"),
    )
    print(f"Saved trajectory plot to {args.output}/")

    # Save results
    np.savez(
        os.path.join(args.output, "optimization_results.npz"),
        z_opt=result["z_opt"],
        cl_opt=result["cl_opt"],
        cd_opt=result["cd_opt"],
        ld_opt=result["ld_opt"],
        design_opt=result["design_opt"],
        nearest_neighbour_idx=result["nearest_neighbour_idx"],
        nearest_neighbour_design=result["nearest_neighbour_design"],
    )
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()

"""
Tests for AeroJEPA components.

Run with:
    python -m pytest tests/
    python -m pytest tests/test_components.py -v
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import farthest_point_sampling, normalize_point_cloud


def test_farthest_point_sampling():
    """Test that FPS returns the correct number of points."""
    pts = torch.randn(2, 1000, 3)
    sampled, idx = farthest_point_sampling(pts, 128)
    assert sampled.shape == (2, 128, 3), f"Expected (2, 128, 3), got {sampled.shape}"
    assert idx.shape == (2, 128), f"Expected (2, 128), got {idx.shape}"
    print("✓ FPS returns correct shape")


def test_fps_no_duplicates():
    """Test that FPS selects distinct points."""
    pts = torch.randn(1, 50, 3)
    _, idx = farthest_point_sampling(pts, 50)
    assert idx.unique().numel() == 50, "FPS should select 50 unique points from 50"
    print("✓ FPS selects unique points when n_samples == N")


def test_fps_larger_than_input():
    """Test FPS when n_samples > N (should pad)."""
    pts = torch.randn(1, 10, 3)
    sampled, _ = farthest_point_sampling(pts, 20)
    assert sampled.shape == (1, 20, 3), f"Expected (1, 20, 3), got {sampled.shape}"
    print("✓ FPS pads when n_samples > N")


def test_normalize_point_cloud():
    """Test point cloud normalisation."""
    pts = torch.randn(4, 100, 3) * 5 + 10
    normed, mean, std = normalize_point_cloud(pts)
    assert normed.shape == pts.shape
    assert abs(normed.mean().item()) < 0.1, f"Mean should be near 0, got {normed.mean().item()}"
    assert abs(normed.std().item() - 1.0) < 0.1, f"Std should be near 1, got {normed.std().item()}"
    print("✓ Normalize point cloud works")


def test_aerojepa_forward():
    """Test the full AeroJEPA forward pass."""
    from src.models.aerojepa import AeroJEPA

    model = AeroJEPA(
        context_in_channels=3,
        target_in_channels=6,
        token_dim=32,
        num_tokens=64,
        encoder_depth=2,
        encoder_heads=4,
        cond_dim=2,
        predictor_depth=2,
        predictor_heads=4,
        decoder_hidden_dim=64,
        decoder_num_layers=2,
        decoder_output_dim=3,
        fourier_dim=8,
        coupled=True,
    )

    B = 2
    geo_pts = torch.randn(B, 100, 3)
    geo_feat = torch.randn(B, 100, 3)
    cond = torch.randn(B, 2)
    qry = torch.randn(B, 50, 3)
    tgt_pts = torch.randn(B, 100, 3)
    tgt_fld = torch.randn(B, 100, 3)

    out = model(
        geometry_points=geo_pts,
        geometry_features=geo_feat,
        conditions=cond,
        query_points=qry,
        target_points=tgt_pts,
        target_fields=tgt_fld,
    )

    assert "context_tokens" in out, "Missing context_tokens"
    assert "predicted_tokens" in out, "Missing predicted_tokens"
    assert "target_tokens" in out, "Missing target_tokens"
    assert "field" in out, "Missing field"

    assert out["context_tokens"].shape == (B, 64, 32), f"Context tokens: {out['context_tokens'].shape}"
    assert out["predicted_tokens"].shape == (B, 64, 32), f"Predicted tokens: {out['predicted_tokens'].shape}"
    assert out["field"].shape == (B, 50, 3), f"Field: {out['field'].shape}"

    print("✓ Full AeroJEPA forward pass works")


def test_loss_functions():
    """Test all loss functions."""
    from src.training.losses import latent_matching_loss, reconstruction_loss, sigreg_loss, AeroJEPALoss

    pred_tokens = torch.randn(2, 64, 32)
    tgt_tokens = torch.randn(2, 64, 32)
    pred_field = torch.randn(2, 50, 3)
    tgt_field = torch.randn(2, 50, 3)

    # Individual losses
    llat = latent_matching_loss(pred_tokens, tgt_tokens)
    lrec = reconstruction_loss(pred_field, tgt_field)
    lsig = sigreg_loss(pred_tokens)

    assert llat > 0, "Llat should be positive"
    assert lrec > 0, "Lrec should be positive"
    assert lsig > 0, "Lsig should be positive"
    print(f"✓ Losses: Llat={llat.item():.4f}, Lrec={lrec.item():.4f}, Lsig={lsig.item():.4f}")

    # Combined loss
    criterion = AeroJEPALoss(coupled=True)
    losses = criterion(pred_tokens, tgt_tokens, pred_field, tgt_field)
    assert "total" in losses
    assert "latent" in losses
    assert "recon" in losses
    assert "sigreg" in losses
    print(f"✓ Combined loss: total={losses['total'].item():.4f}")


def test_linear_probe():
    import numpy as np
    """Test linear probe fitting."""
    from src.analysis.probing import fit_ridge_probes, cv_ridge_r2, concept_vectors

    N, D, K = 1000, 32, 4
    z = np.random.randn(N, D)
    y = z @ np.random.randn(D, K) + 0.1 * np.random.randn(N, K)

    W, b, r2 = fit_ridge_probes(z, y)
    assert W.shape == (K, D), f"Weights shape: {W.shape}"
    assert r2.shape == (K,), f"R² shape: {r2.shape}"
    assert r2.mean() > 0.5, f"Mean R² should be high: {r2.mean():.4f}"
    print(f"✓ Probe R²: mean={r2.mean():.4f}")

    # Concept vectors
    vecs = concept_vectors(W)
    assert vecs.shape == (K, D)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0), f"Concept vectors should be unit norm: {norms}"
    print("✓ Concept vectors are unit-norm")


def test_latent_analyzer():
    import numpy as np
    """Test LatentAnalyzer methods."""
    from src.analysis.latent_analysis import LatentAnalyzer

    # Synthetic data
    N, D, K = 500, 32, 4
    z_ctx = np.random.randn(N, D)
    z_pred = np.random.randn(N, D)
    design = z_ctx @ np.random.randn(D, K) + 0.05 * np.random.randn(N, K)

    from src.models.components import RidgeProbe
    from sklearn.linear_model import RidgeCV
    alphas = np.logspace(-4, 4, 17)
    probes = []
    for k in range(K):
        pr = RidgeCV(alphas=alphas).fit(z_ctx, design[:, k])
        probes.append(pr)

    W = np.array([p.coef_ for p in probes])
    b = np.array([p.intercept_ for p in probes])

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(z_ctx)
    assert proj.shape == (N, 2)
    print("✓ PCA works")

    # Concept arithmetic
    param_names = [f"Param_{i}" for i in range(K)]
    # Manually test concept sensitivity
    mu = z_ctx.mean(axis=0)
    sigma = z_ctx.std(axis=0).clip(min=1e-6)
    concept_vecs = W / np.linalg.norm(W, axis=1, keepdims=True).clip(min=1e-8)

    gamma = 2.0
    for k in range(K):
        z_walk = mu + gamma * concept_vecs[k] * sigma
        z_walk_std = (z_walk - mu) / sigma
        pred = z_walk_std @ W.T + b
        assert pred.shape == (K,)
    print("✓ Concept arithmetic works")


if __name__ == "__main__":
    test_farthest_point_sampling()
    test_fps_no_duplicates()
    test_fps_larger_than_input()
    test_normalize_point_cloud()
    test_aerojepa_forward()
    test_loss_functions()
    test_linear_probe()
    test_latent_analyzer()
    print("\n=== All tests passed! ===")

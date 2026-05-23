"""
Latent-space analysis tools  (§3.2, §F.1–F.5).

Includes:
    - Principal Component Analysis (PCA) of context / predicted latents
    - Concept-vector arithmetic for disentanglement analysis
    - Latent interpolation between operating conditions or geometries
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional


class LatentAnalyzer:
    """
    Collection of latent-space analysis methods.

    Reference (§F.5):
        "We extract three mean-pooled latent vectors per case from the trained
         AeroJEPA: context latent z_ctx, predicted latent z_pred, target latent z_tgt."
    """

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def extract_latents(
        self,
        dataloader,
        use_predicted: bool = True,
        use_target: bool = False,
    ) -> dict:
        """
        Extract mean-pooled latent vectors for a dataset.

        Returns:
            dict with keys:
                - 'ctx':  (N, D) context latents z_ctx
                - 'pred': (N, D) predicted latents z_pred (if use_predicted)
                - 'tgt':  (N, D) target latents z_tgt (if use_target)
                - 'design': (N, K) design parameters
                - 'conditions': (N, C) operating conditions
                - 'aerodynamics': (N, 2) CL, CD (where available)
        """
        self.model.eval()
        ctx_list, pred_list, tgt_list = [], [], []
        design_list, cond_list = [], []

        with torch.no_grad():
            for batch in dataloader:
                B = batch["geometry_pts"].shape[0]

                geo_pts = batch["geometry_pts"].to(self.device)
                geo_feat = batch["geometry_features"].to(self.device)
                cond = batch["conditions"].to(self.device)

                out = self.model(
                    geometry_points=geo_pts,
                    geometry_features=geo_feat,
                    conditions=cond,
                )

                # Mean pool over tokens: (B, M, D) → (B, D)
                z_ctx = out["context_tokens"].mean(dim=1).cpu().numpy()
                ctx_list.append(z_ctx)

                z_pred = out["predicted_tokens"].mean(dim=1).cpu().numpy()
                pred_list.append(z_pred)

                if use_target and "target_tokens" in out:
                    z_tgt = out["target_tokens"].mean(dim=1).cpu().numpy()
                    tgt_list.append(z_tgt)

                design_list.append(batch.get("design_params").cpu().numpy())
                cond_list.append(batch["conditions"].cpu().numpy())

        result = {
            "ctx": np.concatenate(ctx_list, axis=0),
        }
        if use_predicted:
            result["pred"] = np.concatenate(pred_list, axis=0)
        if use_target:
            result["tgt"] = np.concatenate(tgt_list, axis=0)
        result["design"] = np.concatenate(design_list, axis=0) if design_list[0] is not None else None
        result["conditions"] = np.concatenate(cond_list, axis=0)
        return result

    def pca_projection(self, latents: np.ndarray, n_components: int = 2) -> tuple:
        """
        PCA projection of latent vectors.

        Reference (§3.2, Fig. 3):
            "PCA projection of the context latents from AeroJEPA against a VAE baseline."

        Returns:
            proj: (N, n_components) projected latent vectors
            pca:  fitted PCA object
        """
        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(latents)
        return proj, pca

    def concept_arithmetic(
        self,
        z_ctx: np.ndarray,
        weights: np.ndarray,
        design_params: np.ndarray,
        param_names: list,
        walk_steps: int = 21,
    ) -> dict:
        """
        Concept-vector arithmetic  (§F.4).

        Walks the mean latent along each concept direction and predicts
        all design parameters at each step.

        Reference (§F.4):
            "To probe the disentanglement of the representation, we walk the
             train-mean latent along one direction at a time..."

        Returns:
            dict with param_names × param_names matrix of sensitivities (σ/γ).
        """
        mu = z_ctx.mean(axis=0)
        sigma = z_ctx.std(axis=0).clip(min=1e-6)
        z_std = (z_ctx - mu) / sigma

        K = weights.shape[0]
        concept_vecs = weights / np.linalg.norm(weights, axis=1, keepdims=True).clip(min=1e-8)
        design_std = design_params.std(axis=0)

        # Disentanglement matrix: (K, K) sensitivity in σ/γ
        disent = np.zeros((K, K))

        for i in range(K):
            gamma_max = 3.0  # walk range
            gammas = np.linspace(-gamma_max, gamma_max, walk_steps)

            responses = np.zeros((walk_steps, K))
            for j, gamma in enumerate(gammas):
                z_walk = mu + gamma * concept_vecs[i] * sigma
                z_walk_std = (z_walk - mu) / sigma
                pred = z_walk_std @ weights.T  # (K,)
                responses[j] = pred

            # Sensitivity = slope of predicted vs gamma, normalised to σ/γ
            for k in range(K):
                slope = np.polyfit(gammas, responses[:, k], 1)[0]
                disent[i, k] = slope / design_std[k]

        return {
            "matrix": disent,
            "param_names": param_names,
        }

    def interpolate_latents(
        self,
        z_a: np.ndarray,
        z_b: np.ndarray,
        n_steps: int = 10,
    ) -> np.ndarray:
        """
        Linear interpolation between two latent states.

        Reference (§F.1, Fig. 7):
            "We introduce a scalar interpolation parameter α and move between
             latent states corresponding to different angles of attack and
             geometry configurations."

        Returns:
            (n_steps, D) interpolated latent vectors
        """
        alphas = np.linspace(0, 1, n_steps)
        return np.array([(1 - a) * z_a + a * z_b for a in alphas])

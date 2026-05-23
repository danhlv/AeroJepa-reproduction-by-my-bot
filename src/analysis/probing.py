"""
Linear-probe analysis of the AeroJEPA latent space  (§F.5).

Implements the ridge-regression probe used to read out design parameters and
aerodynamic coefficients from the self-supervised latent representations.

Reference (§F.5):
    "For a latent vector z ∈ R^D and a scalar target y, we define a linear probe
     as the composition of train-set standardisation and ridge regression with
     cross-validated regularisation."
"""

import torch
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from typing import Optional


def fit_ridge_probes(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    alphas: list = None,
) -> tuple:
    """
    Fit ridge probes from latent vectors to targets.

    Args:
        z_train: (N_train, D) latent vectors
        y_train: (N_train, K) target values
        z_val:   (N_val, D) optional validation latent vectors
        y_val:   (N_val, K) optional validation targets
        alphas:  regularisation strengths

    Returns:
        weights: (K, D) probe weights
        biases:  (K,)    probe biases
        r2:      (K,)    R² scores (on validation if provided, else training)
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 17)

    K = y_train.shape[1]
    weights = np.zeros((K, z_train.shape[1]))
    biases = np.zeros(K)
    r2 = np.zeros(K)

    for k in range(K):
        probe = RidgeCV(alphas=alphas, scoring="r2")
        probe.fit(z_train, y_train[:, k])

        weights[k] = probe.coef_
        biases[k] = probe.intercept_

        if z_val is not None and y_val is not None:
            y_pred = probe.predict(z_val)
            r2[k] = r2_score(y_val[:, k], y_pred)
        else:
            y_pred = probe.predict(z_train)
            r2[k] = r2_score(y_train[:, k], y_pred)

    return weights, biases, r2


def cv_ridge_r2(
    z: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    alphas: list = None,
) -> np.ndarray:
    """
    Cross-validated R² for ridge probes.

    Args:
        z:      (N, D) latent vectors
        y:      (N, K) targets
        n_folds: number of CV folds
        alphas:  regularisation strengths

    Returns:
        r2_scores: (K,) mean CV R² per target
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge

    if alphas is None:
        alphas = np.logspace(-4, 4, 17)

    N, K = y.shape
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = np.zeros(K)

    for k in range(K):
        fold_r2 = []
        for train_idx, val_idx in kf.split(z):
            z_tr, z_val_ = z[train_idx], z[val_idx]
            y_tr, y_val_ = y[train_idx, k], y[val_idx, k]

            # Inner CV for alpha selection
            inner_probe = RidgeCV(alphas=alphas, scoring="r2")
            inner_probe.fit(z_tr, y_tr)

            # Evaluate on held-out fold
            y_pred = inner_probe.predict(z_val_)
            fold_r2.append(r2_score(y_val_, y_pred))

        r2_scores[k] = np.mean(fold_r2)

    return r2_scores


def concept_vectors(weights: np.ndarray) -> np.ndarray:
    """
    Compute unit-norm concept vectors from probe weight matrix (§F.4).

    Reference (§F.5):
        "the unit-norm probe direction v_k = w_k / ||w_k|| is interpreted as
         the latent concept vector for parameter x_k"

    Args:
        weights: (K, D) probe weight matrix

    Returns:
        concepts: (K, D) unit-norm concept vectors
    """
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return weights / norms


class LinearProbe:
    """PyTorch-compatible linear probe for differentiable latent analysis."""

    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        self.W = None
        self.b = None
        self.mu = None
        self.sigma = None

    def fit(self, z: np.ndarray, y: np.ndarray):
        self.mu = np.mean(z, axis=0, keepdims=True)
        self.sigma = np.std(z, axis=0, keepdims=True).clip(min=1e-6)
        z_std = (z - self.mu) / self.sigma
        self.W, self.b, self.r2 = fit_ridge_probes(z_std, y)

    def predict_numpy(self, z: np.ndarray) -> np.ndarray:
        z_std = (z - self.mu) / self.sigma
        return z_std @ self.W.T + self.b

    def to_torch(self, device="cpu") -> tuple:
        """Return (W, b, mu, sigma) as torch tensors for autograd."""
        return (
            torch.from_numpy(self.W).float().to(device),
            torch.from_numpy(self.b).float().to(device),
            torch.from_numpy(self.mu).float().to(device),
            torch.from_numpy(self.sigma).float().to(device),
        )

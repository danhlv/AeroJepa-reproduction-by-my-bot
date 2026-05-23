"""
Constraint utilities for latent-space optimisation  (§F.3).

Implements:
    - Mahalanobis trust region (keeps search on the latent manifold)
    - Design bound constraints (affine inequalities on decodeable parameters)
"""

import numpy as np
from scipy.stats import chi2


class MahalanobisTrustRegion:
    """
    Mahalanobis trust region constraint  (§F.3, constraint 1).

    Reference (§F.3):
        "A Mahalanobis trust region (z_ctx − μ_train)ᵀ Σ⁻¹_train (z_ctx − μ_train) ≤ τ
         that keeps the search on the latent manifold."
    """

    def __init__(self, z_train: np.ndarray, threshold_quantile: float = 0.95):
        self.mu = z_train.mean(axis=0)
        self.cov = np.cov(z_train, rowvar=False)
        self.inv_cov = np.linalg.inv(self.cov + 1e-8 * np.eye(z_train.shape[1]))
        self.threshold = chi2.ppf(threshold_quantile, df=z_train.shape[1])

    def distance(self, z: np.ndarray) -> float:
        """Compute Mahalanobis distance."""
        delta = z - self.mu
        return np.sqrt(delta @ self.inv_cov @ delta)

    def constraint(self, z: np.ndarray) -> float:
        """Inequality constraint: ≤ 0 means inside trust region."""
        d2 = self.distance(z) ** 2
        return self.threshold - d2

    def constraint_jac(self, z: np.ndarray) -> np.ndarray:
        """Jacobian of the constraint."""
        delta = z - self.mu
        grad = -2 * self.inv_cov @ delta
        return grad

    def project_inside(self, z: np.ndarray) -> np.ndarray:
        """Project a point inside the trust region by scaling."""
        d2 = (z - self.mu) @ self.inv_cov @ (z - self.mu)
        if d2 > self.threshold:
            scale = np.sqrt(self.threshold / d2)
            return self.mu + scale * (z - self.mu)
        return z


class DesignBoundConstraint:
    """
    Bounds on linearly decodable design parameters  (§F.3, constraint 2).

    Reference (§F.3):
        "Bounds on the design parameters that are reliably linearly decodable
         from z_ctx (5-fold CV R² ≥ 0.85, retaining 9 of the 54 SuperWing
         parameters)."
    """

    def __init__(
        self,
        weights: np.ndarray,     # (K, D) probe weights
        biases: np.ndarray,      # (K,) probe biases
        x_min: np.ndarray,       # (K,) lower bounds
        x_max: np.ndarray,       # (K,) upper bounds
        r2_scores: np.ndarray,   # (K,) R² scores
        r2_threshold: float = 0.85,
    ):
        # Keep only high-quality probes
        self.valid = r2_scores >= r2_threshold
        if not self.valid.any():
            raise ValueError("No probes meet the R² threshold.")

        self.weights = weights[self.valid]
        self.biases = biases[self.valid]
        self.x_min = x_min[self.valid]
        self.x_max = x_max[self.valid]
        self.K = self.valid.sum()

    def decode(self, z_std: np.ndarray) -> np.ndarray:
        """Decode design parameters from standardised latent."""
        return z_std @ self.weights.T + self.biases

    def constraints(self, z: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> list:
        """
        Return a list of dict-style constraints for scipy.optimize.
        x_min ≤ decode(z_std) ≤ x_max
        """
        z_std = (z - mu) / sigma.clip(min=1e-6)
        x = self.decode(z_std)

        cons = []
        for k in range(self.K):
            cons.append({
                "type": "ineq",
                "fun": lambda z, kk=k: x[kk] - self.x_min[kk],
                "jac": lambda z, kk=k: self.weights[kk] / sigma.clip(min=1e-6),
            })
            cons.append({
                "type": "ineq",
                "fun": lambda z, kk=k: self.x_max[kk] - x[kk],
                "jac": lambda z, kk=k: -self.weights[kk] / sigma.clip(min=1e-6),
            })
        return cons

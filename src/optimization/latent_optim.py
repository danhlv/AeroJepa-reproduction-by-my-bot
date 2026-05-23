"""
Constrained optimisation in latent space  (§F.3).

Implements the proof-of-concept design optimisation using SLSQP with:
    - Mahalanobis trust region (Eq. 9, constraint 1)
    - Design parameter bounds (constraint 2)
    - Aerodynamic floors / ceilings (constraints 3–4)

Reference (§F.3):
    "We solve min_{z_ctx} -CL/CD with SLSQP, subject to four families of
     physically motivated guardrails, all derived from training-set statistics."
"""

import torch
import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Optional, Callable


class LatentOptimizer:
    """
    Latent-space aerodynamic optimisation  (§F.3, Fig. 4, 14).

    Optimises CL/CD by searching directly in the context latent space using
    a differentiable chain: context latent → predictor → linear probes.

    Reference (§F.3):
        "The frozen AeroJEPA predictor maps a context latent and a flow condition
         to a fluid latent, and two linear ridge probes read out CL and CD."
    """

    def __init__(
        self,
        predictor,              # frozen AeroJEPA predictor
        design_probe_weights: np.ndarray,
        design_probe_biases: np.ndarray,
        cl_probe_weights: np.ndarray,
        cd_probe_weights: np.ndarray,
        cl_probe_bias: float,
        cd_probe_bias: float,
        mu_ctx: np.ndarray,
        sigma_ctx: np.ndarray,
        train_design_params: np.ndarray,
        train_cl: np.ndarray,
        train_cd: np.ndarray,
        config,
        device: str = "cpu",
    ):
        self.predictor = predictor
        self.device = device

        # Probes (as torch tensors for autograd)
        self.W_design = torch.from_numpy(design_probe_weights).float().to(device)
        self.b_design = torch.from_numpy(design_probe_biases).float().to(device)
        self.w_cl = torch.from_numpy(cl_probe_weights).float().to(device)
        self.w_cd = torch.from_numpy(cd_probe_weights).float().to(device)
        self.b_cl = torch.tensor(cl_probe_bias, device=device)
        self.b_cd = torch.tensor(cd_probe_bias, device=device)

        # Standardisation
        self.mu_ctx = torch.from_numpy(mu_ctx).float().to(device)
        self.sigma_ctx = torch.from_numpy(sigma_ctx).float().to(device).clamp(min=1e-6)

        # Training statistics for constraints
        self.train_cl = train_cl
        self.train_cd = train_cd
        self.train_design = train_design_params
        self.config = config

        # Mahalanobis
        self.mahalanobis_cov = np.cov(train_design_params, rowvar=False)
        self.mahalanobis_mu = train_design_params.mean(axis=0)
        # Use Chi2 threshold for trust region
        from scipy.stats import chi2
        self.mahalanobis_thresh = chi2.ppf(config.optim_mahalanobis_threshold, df=128)

    def _standardize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mu_ctx) / self.sigma_ctx

    def _predict_aero(self, z_ctx: torch.Tensor) -> tuple:
        """Predict CL and CD from context latent."""
        z_std = self._standardize(z_ctx)
        # SLSQP passes 1D; expand for predictor
        if z_ctx.dim() == 1:
            z_ctx_batch = z_ctx.unsqueeze(0)
        else:
            z_ctx_batch = z_ctx

        with torch.set_grad_enabled(True):
            # Through predictor
            z_pred = self.predictor(
                context_tokens=z_ctx_batch.unsqueeze(1),  # add token dim (single-token approx)
                conditions=torch.zeros(z_ctx_batch.shape[0], self.config.condition_dim,
                                       device=self.device),  # fixed cruise condition
            ).squeeze(1)

            # Linear probes
            z_std_pred = (z_pred - z_pred.mean()) / z_pred.std().clamp(min=1e-6)
            cl = (z_std_pred @ self.w_cl + self.b_cl).squeeze()
            cd = (z_std_pred @ self.w_cd + self.b_cd).squeeze()

        return cl, cd

    def _objective_and_grad(self, z: np.ndarray) -> tuple:
        """Compute -CL/CD and its gradient w.r.t. z."""
        z_t = torch.from_numpy(z).float().to(self.device).requires_grad_(True)
        cl, cd = self._predict_aero(z_t)
        obj = -cl / cd
        obj.backward()
        grad = z_t.grad.cpu().numpy().copy()
        return obj.item(), grad

    def _constraints(self, z: np.ndarray) -> dict:
        """Compute inequality constraints."""
        cl, cd = self._predict_aero(torch.from_numpy(z).float().to(self.device))

        constraints = []
        # Constraint 3: drag floor and lift ceiling
        drag_floor = 0.9 * self.train_cd.min()
        lift_ceiling = 1.05 * self.train_cl.max()
        constraints.append({"type": "ineq", "fun": lambda z: cd.item() - drag_floor})
        constraints.append({"type": "ineq", "fun": lambda z: lift_ceiling - cl.item()})

        # Constraint 4: L/D ceiling
        ld_ceiling = (self.train_cl / self.train_cd).max()
        constraints.append({
            "type": "ineq",
            "fun": lambda z: ld_ceiling - cl.item() / cd.item(),
        })

        return constraints

    def optimize(
        self,
        condition: np.ndarray,
        n_restarts: int = 8,
    ) -> dict:
        """
        Run constrained latent-space optimisation.

        Reference (§F.3):
            "We use eight random restarts, drawing initial z_ctx values from the
             training distribution and projecting them inside the Mahalanobis ball."

        Args:
            condition: (C,) operating condition for optimisation
            n_restarts: number of random restarts

        Returns:
            dict with optimum z, design params, CL, CD, L/D, and nearest-neighbour
        """
        best_result = None
        best_obj = float("inf")

        for restart in range(n_restarts):
            # Sample initial point from training distribution
            z0 = self.mu_ctx.cpu().numpy() + \
                 0.1 * self.sigma_ctx.cpu().numpy() * np.random.randn(128)

            # SLSQP optimisation
            res = minimize(
                self._objective_and_grad,
                z0,
                method="SLSQP",
                jac=True,
                constraints=self._constraints(z0),
                options={"maxiter": 200, "ftol": 1e-8},
            )

            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res

        # Decode optimum
        z_opt = best_result.x
        cl_opt, cd_opt = self._predict_aero(
            torch.from_numpy(z_opt).float().to(self.device)
        )
        ld_opt = cl_opt / cd_opt

        # Map back to design space
        z_std_opt = self._standardize(torch.from_numpy(z_opt).float().to(self.device))
        design_opt = (z_std_opt @ self.W_design.T + self.b_design).cpu().numpy()

        # Nearest neighbour in design space
        nn_idx = np.argmin(
            np.sum((self.train_design - design_opt) ** 2, axis=1)
        )

        return {
            "z_opt": z_opt,
            "cl_opt": cl_opt.item(),
            "cd_opt": cd_opt.item(),
            "ld_opt": ld_opt.item(),
            "design_opt": design_opt.flatten(),
            "nearest_neighbour_idx": nn_idx,
            "nearest_neighbour_design": self.train_design[nn_idx],
            "n_restarts": n_restarts,
            "success": best_result.success,
        }

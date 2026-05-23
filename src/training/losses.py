"""
Training objectives  (§2.6).

Implements:
    - Llat: latent matching loss (MSE between Ẑt and Zt)
    - Lrec: reconstruction loss (MSE on decoded field at query points)
    - Lsig: SIGReg regularisation (prevents representation collapse)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def latent_matching_loss(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Llat — latent matching loss  (Eq. 7).

    Reference (§2.6):
        "Llat = ||Ẑt - Zt||²₂ which aligns the predictor output with the
         target encoder output token by token."

    Args:
        predicted_tokens: (B, M, d) Ẑt
        target_tokens:    (B, M, d) Zt
    Returns:
        scalar loss
    """
    return F.mse_loss(predicted_tokens, target_tokens)


def reconstruction_loss(
    predicted_field: torch.Tensor,
    target_field: torch.Tensor,
) -> torch.Tensor:
    """
    Lrec — reconstruction loss  (Eq. 8).

    Reference (§2.6):
        "Lrec = E_{q∈Ω}[||f_dec(Ẑt, q) − F(q)||²₂]"

    Args:
        predicted_field: (B, Nq, C) decoded field values
        target_field:    (B, Nq, C) ground-truth field values
    Returns:
        scalar loss
    """
    return F.mse_loss(predicted_field, target_field)


def sigreg_loss(
    tokens: torch.Tensor,
    n_projections: int = 16,
    projection_dim: int = 16,
) -> torch.Tensor:
    """
    Lsig — SIGReg regulariser  (§2.6).

    Prevents representation collapse by regularising the latent distribution
    toward an isotropic Gaussian via random low-dimensional projections.

    Reference (§2.6):
        "SIGReg applies regularizes the latent distribution through random
         low-dimensional projections toward an isotropic Gaussian prior."
        "SIGReg [Balestriero and LeCun, 2025, Maes et al., 2026]."

    This is a simplified version.  The full SIGReg implementation follows
    Balestriero & LeCun (2025) and Maes et al. (2026).
    """
    B, M, d = tokens.shape
    device = tokens.device

    # Random projections
    with torch.no_grad():
        proj = torch.randn(d, n_projections, projection_dim, device=device)
        proj = proj / proj.norm(dim=0, keepdim=True).clamp(min=1e-8)

    # Project tokens into low-dimensional space: (B, M, n_proj, proj_dim)
    token_proj = torch.einsum("bmd,dpk->bmpk", tokens, proj)

    # Target: standard Gaussian
    target = torch.randn_like(token_proj)

    # Collapse-avoidance: encourage token means to be zero and variances to be one
    mean_proj = token_proj.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1, proj_dim)
    var_proj = token_proj.var(dim=(1, 2), keepdim=True).clamp(min=1e-8)

    loss = (
        0.5 * (token_proj - target).pow(2).mean()
        + 0.1 * (1 - var_proj).abs().mean()  # variance regularisation
    )
    return loss


class AeroJEPALoss(nn.Module):
    """
    Combined AeroJEPA loss  (Eq. 5, 6).

    L_total = λℓ * Llat + λr * Lrec + λs * Lsig
    or
    L_latent-only = λℓ * Llat + λs * Lsig

    Usage:
        criterion = AeroJEPALoss(lambda_latent=1.0, lambda_recon=1.0, lambda_sigreg=0.01)
        losses = criterion(pred_tokens, target_tokens, pred_field, target_field)
    """

    def __init__(
        self,
        lambda_latent: float = 1.0,
        lambda_recon: float = 1.0,
        lambda_sigreg: float = 0.01,
        coupled: bool = True,
    ):
        super().__init__()
        self.lambda_latent = lambda_latent
        self.lambda_recon = lambda_recon
        self.lambda_sigreg = lambda_sigreg
        self.coupled = coupled

    def forward(
        self,
        predicted_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        predicted_field: torch.Tensor = None,
        target_field: torch.Tensor = None,
    ) -> dict:
        """
        Returns:
            dict with 'total', 'latent', 'recon', 'sigreg' loss scalars.
        """
        losses = {}

        # Latent matching (always)
        losses["latent"] = latent_matching_loss(predicted_tokens, target_tokens)

        # SIGReg on predicted tokens (always)
        losses["sigreg"] = sigreg_loss(predicted_tokens)

        # Reconstruction (only in coupled mode)
        if self.coupled and predicted_field is not None and target_field is not None:
            losses["recon"] = reconstruction_loss(predicted_field, target_field)
        else:
            losses["recon"] = torch.tensor(0.0, device=predicted_tokens.device)

        # Total
        losses["total"] = (
            self.lambda_latent * losses["latent"]
            + self.lambda_recon * losses["recon"]
            + self.lambda_sigreg * losses["sigreg"]
        )

        return losses

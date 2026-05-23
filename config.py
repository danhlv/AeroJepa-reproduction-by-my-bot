"""
AeroJEPA: Learning Semantic Latent Representations for Scalable 3D Aerodynamic Field Modeling
==============================================================================================
Central configuration. See Table 3 of the paper for HiLiftAeroML vs SuperWing defaults.

Usage:
    from config import AeroJEPAConfig
    cfg = AeroJEPAConfig(dataset='superwing')
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Dataset-specific defaults  (Table 3 in the paper)
# ---------------------------------------------------------------------------
_HILIFT_DEFAULTS = {
    "nc": 131_072,
    "nt": 131_072,
    "nq": 131_072,
    "m_tokens": 3072,
    "d_token": 64,
    "encoder_depth": 6,
    "predictor_depth": 6,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "grad_clip": 10.0,
    "epochs": 300,
    "scheduler": "cosine_warmup",
    "warmup_steps": 1000,
    "volumetric": False,
    "surface_targets": ["u", "v", "w", "p"],
    "num_design_params": 8,
    "condition_dim": 1,          # angle of attack only
}

_SUPERWING_DEFAULTS = {
    "nc": 8_192,
    "nt": 8_192,
    "nq": 8_192,
    "m_tokens": 512,
    "d_token": 128,
    "encoder_depth": 6,
    "predictor_depth": 6,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "grad_clip": 10.0,
    "epochs": 200,
    "scheduler": "cosine_warmup",
    "warmup_steps": 1000,
    "volumetric": False,
    "surface_targets": ["Cp", "Cf_tau", "Cf_z"],
    "num_design_params": 54,
    "condition_dim": 2,          # alpha, Mach
}


def _defaults_for(dataset: str) -> dict:
    ds = dataset.lower().replace("-", "").replace("_", "")
    if ds in ("hiliftaeroml", "hilift"):
        return dict(_HILIFT_DEFAULTS)
    elif ds in ("superwing", "wing"):
        return dict(_SUPERWING_DEFAULTS)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.  Choose 'hiliftaeroml' or 'superwing'.")


@dataclass
class AeroJEPAConfig:
    """Top-level configuration dataclass."""

    # ---------- dataset ----------
    dataset: str = "superwing"            # "hiliftaeroml" | "superwing"
    data_root: str = "./data"
    output_dir: str = "./outputs"

    # ---------- sampling ----------
    nc: int = 8_192                       # geometry context subsample
    nt: int = 8_192                       # fluid target subsample
    nq: int = 8_192                       # reconstruction query set

    # ---------- latent tokenisation ----------
    m_tokens: int = 512                   # number of spatial tokens (M)
    d_token: int = 128                    # token feature dimension (d)
    num_neighbours: int = 16              # neighbourhood size for message-passing aggregation
    fps_scale: float = 2.0                # FPS centroid count = m_tokens * fps_scale

    # ---------- encoders ----------
    encoder_depth: int = 6                # transformer blocks
    encoder_heads: int = 8
    encoder_mlp_ratio: float = 4.0
    encoder_dropout: float = 0.0
    use_sdf: bool = False                 # provide signed-distance values for volumetric

    # ---------- predictor ----------
    predictor_depth: int = 6
    predictor_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    condition_dim: int = 2                # alpha, Mach (SuperWing)  — 1 for HiLift (AoA)
    fourier_dim: int = 32                 # Fourier feature encoding size for query coordinates

    # ---------- decoder (INR) ----------
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 4
    decoder_use_sdf: bool = False
    fourier_scale: float = 10.0            # scale for Fourier feature encoding

    # ---------- training ----------
    lr: float = 1e-3
    weight_decay: float = 1e-3
    grad_clip: float = 10.0
    epochs: int = 200
    scheduler: str = "cosine_warmup"
    warmup_steps: int = 1000
    batch_size: int = 8
    seed: int = 42

    # ---------- loss weights (Eq. 5) ----------
    lambda_latent: float = 1.0           # λℓ
    lambda_recon: float = 1.0            # λr
    lambda_sigreg: float = 0.01          # λs

    # ---------- training mode ----------
    coupled: bool = True                  # True = end-to-end, False = decoupled latent-first
    decoupled_decoder_epochs: int = 50    # decoder-only training when coupled=False

    # ---------- checkpointing ----------
    checkpoint_every: int = 10
    resume: Optional[str] = None

    # ---------- optimisation (latent-space) ----------
    optim_mahalanobis_threshold: float = 0.95     # trust-region quantile
    optim_max_restarts: int = 8
    optim_design_probe_r2_threshold: float = 0.85
    optim_lift_ceiling_factor: float = 1.05
    optim_drag_floor_factor: float = 0.9

    def __post_init__(self):
        # Apply dataset-specific defaults, overriding only if the user left defaults
        ds_defaults = _defaults_for(self.dataset)
        for k, v in ds_defaults.items():
            if hasattr(self, k) and getattr(self, k) in (
                8_192, 131_072, 512, 3072, 64, 128, 6, 1e-3, 1e-3, 10.0,
                200, 300, False,
            ) and k != "d_token":
                # Only override if the current value matches a common default
                # (heuristic: check if we haven't explicitly changed it)
                pass
            # Apply all defaults for explicit dataset matches
            if k == "nc":
                object.__setattr__(self, "nc", v)
            elif k == "nt":
                object.__setattr__(self, "nt", v)
            elif k == "nq":
                object.__setattr__(self, "nq", v)
            elif k == "m_tokens":
                object.__setattr__(self, "m_tokens", v)
            elif k == "d_token":
                object.__setattr__(self, "d_token", v)
            elif k == "encoder_depth":
                object.__setattr__(self, "encoder_depth", v)
            elif k == "lr":
                object.__setattr__(self, "lr", v)
            elif k == "weight_decay":
                object.__setattr__(self, "weight_decay", v)
            elif k == "grad_clip":
                object.__setattr__(self, "grad_clip", v)
            elif k == "epochs":
                object.__setattr__(self, "epochs", v)
            elif k == "condition_dim":
                object.__setattr__(self, "condition_dim", v)
            else:
                object.__setattr__(self, k, v)

        # Derived
        object.__setattr__(self, "device", "cuda" if __import__("torch").cuda.is_available() else "cpu")
        object.__setattr__(self, "token_fps_k", int(self.m_tokens * 2.0))

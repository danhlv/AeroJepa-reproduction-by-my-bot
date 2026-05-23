"""
AeroJEPA: main model  (§2.2).

Combines context encoder, target encoder (training only), latent predictor,
and optional INR decoder into the full AeroJEPA architecture.

Both coupled (end-to-end) and decoupled (latent-first) training are supported.

Reference (§2.2, Fig. 1):
    "The context encoder and predictor are always used at inference.
     The target encoder is used only during training.
     The decoder is optional."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ContextEncoder, TargetEncoder
from .predictor import LatentPredictor
from .decoder import INRDecoder


class AeroJEPA(nn.Module):
    """
    Full AeroJEPA model  (Fig. 1).

    Inference modes:
        - latent_only: Ẑt = predictor(context_encoder(geometry), conditions)
        - decode:      field = decoder(Ẑt, query_points)

    Training modes:
        - coupled:     full forward pass through all modules, both latent and recon losses
        - decoupled:   only encoders + predictor, decoder trained separately later
    """

    def __init__(
        self,
        # Context encoder config
        context_in_channels: int = 3,
        target_in_channels: int = 7,       # e.g., 3 (coords) + 4 (u,v,w,p)
        token_dim: int = 128,
        num_tokens: int = 512,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        # Predictor config
        cond_dim: int = 2,
        predictor_depth: int = 6,
        predictor_heads: int = 8,
        # Decoder config
        decoder_hidden_dim: int = 512,
        decoder_num_layers: int = 4,
        decoder_output_dim: int = 4,
        fourier_dim: int = 32,
        decoder_use_sdf: bool = False,
        # Training mode
        coupled: bool = True,
        # Dropout / regularisation
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.coupled = coupled
        self.token_dim = token_dim
        self.num_tokens = num_tokens

        # ----------------- Encoders -----------------
        self.context_encoder = ContextEncoder(
            in_channels=context_in_channels,
            token_dim=token_dim,
            num_tokens=num_tokens,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.target_encoder = TargetEncoder(
            in_channels=target_in_channels,
            token_dim=token_dim,
            num_tokens=num_tokens,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # ----------------- Predictor -----------------
        self.predictor = LatentPredictor(
            token_dim=token_dim,
            num_tokens=num_tokens,
            cond_dim=cond_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            fourier_dim=fourier_dim,
        )

        # ----------------- Decoder -----------------
        self.decoder = INRDecoder(
            token_dim=token_dim,
            num_tokens=num_tokens,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            output_dim=decoder_output_dim,
            fourier_dim=fourier_dim,
            use_sdf=decoder_use_sdf,
        )

    def forward(
        self,
        geometry_points: torch.Tensor,
        geometry_features: torch.Tensor,
        conditions: torch.Tensor,
        query_points: torch.Tensor = None,
        target_points: torch.Tensor = None,
        target_fields: torch.Tensor = None,
        sdf_values: torch.Tensor = None,
        return_latents: bool = False,
    ) -> dict:
        """
        Forward pass.

        Args:
            geometry_points:    (B, Nc, 3) subsampled geometry coordinates
            geometry_features:  (B, Nc, C_in) geometry point features
            conditions:         (B, C_cond) operating conditions
            query_points:       (B, Nq, 3) query coordinates for decoder (optional)
            target_points:      (B, Nt, 3) flow coordinates (training only)
            target_fields:      (B, Nt, C_flow) flow field values (training only)
            sdf_values:         (B, Nq, 1) SDF at query points (volumetric)
            return_latents:     whether to return intermediate latent tokens

        Returns:
            dict with keys:
                - predicted_tokens: (B, M, d) Ẑt
                - context_tokens:    (B, M, d) Zc
                - target_tokens:     (B, M, d) Zt  (only if target data provided)
                - field:             (B, Nq, C_out) decoded field (only if query provided)
        """
        B = geometry_points.shape[0]

        # ---- Context encoding ----
        zc = self.context_encoder(geometry_points, geometry_features)  # (B, M, d)

        # ---- Latent prediction ----
        z_hat = self.predictor(zc, conditions)  # (B, M, d)

        out = {
            "context_tokens": zc,
            "predicted_tokens": z_hat,
        }

        # ---- Target encoding (training only) ----
        if target_points is not None and target_fields is not None:
            # Concatenate coordinates + field values as encoder input
            target_input = torch.cat([target_points, target_fields], dim=-1)
            zt = self.target_encoder(target_points, target_input)
            out["target_tokens"] = zt

        # ---- Decoding (optional) ----
        if query_points is not None:
            field = self.decoder(z_hat, query_points, sdf_values)
            out["field"] = field

        if return_latents:
            out["predicted_tokens"] = z_hat
            out["context_tokens"] = zc

        return out

    def encode_context(self, geometry_points: torch.Tensor, geometry_features: torch.Tensor) -> torch.Tensor:
        """Encode geometry → context tokens (useful for caching)."""
        return self.context_encoder(geometry_points, geometry_features)

    def predict_latent(self, context_tokens: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Predict latent from cached context tokens."""
        return self.predictor(context_tokens, conditions)

    def decode_field(self, predicted_tokens: torch.Tensor, query_points: torch.Tensor,
                     sdf_values: torch.Tensor = None) -> torch.Tensor:
        """Decode field at arbitrary query locations from predicted tokens."""
        return self.decoder(predicted_tokens, query_points, sdf_values)

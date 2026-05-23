"""
Training loop  (§E.6).

Implements the coupled (end-to-end) and decoupled (latent-first) training
procedures using AdamW with cosine warmup scheduling.

Reference (§E.6):
    "Network weights are updated using the AdamW optimizer, stabilized by a
     gradient clip.  All models were trained on a single NVIDIA H200 GPU."
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
from tqdm import tqdm
from typing import Optional

from .losses import AeroJEPALoss


def cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Cosine learning rate with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class AeroJEPATrainer:
    """
    Trainer for AeroJEPA models.

    Supports:
        - Coupled training (latent + reconstruction, Eq. 5)
        - Decoupled training (latent-only first, then decoder, Eqs. 5 + 6)
        - Gradient clipping (§E.6: "stabilized by a gradient clip")
        - Cosine warmup LR schedule (§E.6)
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device(config.device)
        self.model.to(self.device)

        # Optimiser (§E.6: AdamW)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Loss
        self.criterion = AeroJEPALoss(
            lambda_latent=config.lambda_latent,
            lambda_recon=config.lambda_recon,
            lambda_sigreg=config.lambda_sigreg,
            coupled=config.coupled,
        )

    def _setup_scheduler(self, total_steps: int):
        return cosine_warmup_scheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
        )

    def train_epoch(
        self,
        dataloader,
        epoch: int,
        scheduler=None,
        log_interval: int = 50,
    ) -> dict:
        """Run one training epoch."""
        self.model.train()
        total_losses = {"total": 0.0, "latent": 0.0, "recon": 0.0, "sigreg": 0.0}
        n_batches = len(dataloader)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            # Move batch to device
            geo_pts = batch["geometry_pts"].to(self.device)
            geo_feat = batch["geometry_features"].to(self.device)
            cond = batch["conditions"].to(self.device)
            tgt_pts = batch["flow_pts"].to(self.device)
            tgt_fld = batch["flow_fields"].to(self.device)
            qry_pts = batch["query_pts"].to(self.device)
            qry_fld = batch["query_fields"].to(self.device)

            # Forward
            out = self.model(
                geometry_points=geo_pts,
                geometry_features=geo_feat,
                conditions=cond,
                query_points=qry_pts,
                target_points=tgt_pts,
                target_fields=tgt_fld,
            )

            # Loss
            losses_dict = self.criterion(
                predicted_tokens=out["predicted_tokens"],
                target_tokens=out["target_tokens"],
                predicted_field=out.get("field"),
                target_field=qry_fld,
            )

            # Backward
            self.optimizer.zero_grad()
            losses_dict["total"].backward()

            # Gradient clipping (§E.6)
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Logging
            for k in total_losses:
                total_losses[k] += losses_dict[k].item()
            if step % log_interval == 0:
                pbar.set_postfix({k: f"{v / (step + 1):.4f}" for k, v in total_losses.items()})

        return {k: v / n_batches for k, v in total_losses.items()}

    def train(
        self,
        train_loader,
        val_loader=None,
        epochs: Optional[int] = None,
        output_dir: str = "./outputs",
    ):
        """Full training loop."""
        epochs = epochs or self.config.epochs
        os.makedirs(output_dir, exist_ok=True)

        total_steps = len(train_loader) * epochs
        scheduler = self._setup_scheduler(total_steps)

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch, scheduler)
            print(f"Epoch {epoch}/{epochs} — Train Loss: {train_loss['total']:.4f} "
                  f"(lat={train_loss['latent']:.4f}, rec={train_loss['recon']:.4f}, "
                  f"sig={train_loss['sigreg']:.4f})")

            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"  Val Loss: {val_loss['total']:.4f}")

                # Save best checkpoint
                if val_loss["total"] < best_val_loss:
                    best_val_loss = val_loss["total"]
                    self.save_checkpoint(
                        os.path.join(output_dir, "best.pt"),
                        epoch, val_loss["total"],
                    )

            # Periodic checkpoint
            if epoch % self.config.checkpoint_every == 0:
                self.save_checkpoint(
                    os.path.join(output_dir, f"checkpoint_{epoch:04d}.pt"),
                    epoch, train_loss["total"],
                )

        # Final checkpoint
        self.save_checkpoint(
            os.path.join(output_dir, "final.pt"),
            epochs, train_loss["total"],
        )
        return train_loss

    def validate(self, dataloader) -> dict:
        """Validation loop."""
        self.model.eval()
        total_losses = {"total": 0.0, "latent": 0.0, "recon": 0.0, "sigreg": 0.0}
        n_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                geo_pts = batch["geometry_pts"].to(self.device)
                geo_feat = batch["geometry_features"].to(self.device)
                cond = batch["conditions"].to(self.device)
                tgt_pts = batch["flow_pts"].to(self.device)
                tgt_fld = batch["flow_fields"].to(self.device)
                qry_pts = batch["query_pts"].to(self.device)
                qry_fld = batch["query_fields"].to(self.device)

                out = self.model(
                    geometry_points=geo_pts,
                    geometry_features=geo_feat,
                    conditions=cond,
                    query_points=qry_pts,
                    target_points=tgt_pts,
                    target_fields=tgt_fld,
                )
                losses_dict = self.criterion(
                    predicted_tokens=out["predicted_tokens"],
                    target_tokens=out["target_tokens"],
                    predicted_field=out.get("field"),
                    target_field=qry_fld,
                )
                for k in total_losses:
                    total_losses[k] += losses_dict[k].item()

        return {k: v / n_batches for k, v in total_losses.items()}

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"], ckpt["loss"]

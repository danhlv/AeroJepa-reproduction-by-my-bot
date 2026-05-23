#!/usr/bin/env python3
"""
Training pipeline for AeroJEPA  (§E.6).

Supports:
    - Coupled end-to-end training (latent + reconstruction, Eq. 5)
    - Decoupled latent-first + decoder-only training (Eq. 6)
    - HiLiftAeroML and SuperWing datasets

Usage:
    python pipelines/train_pipeline.py --dataset superwing --coupled
    python pipelines/train_pipeline.py --dataset hiliftaeroml --epochs 300

Reference (§E.6):
    "All models were trained on a single NVIDIA H200 GPU."
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AeroJEPAConfig
from src.data.datasets import HiLiftAeroMLDataset, SuperWingDataset, collate_aerojepa
from src.models.aerojepa import AeroJEPA
from src.training.trainer import AeroJEPATrainer


def build_dataloaders(cfg: AeroJEPAConfig):
    """Create train/val dataloaders for the chosen dataset."""
    if "hilift" in cfg.dataset.lower():
        Dataset = HiLiftAeroMLDataset
    else:
        Dataset = SuperWingDataset

    train_dataset = Dataset(
        root=cfg.data_root,
        split="train",
        nc=cfg.nc,
        nt=cfg.nt,
        nq=cfg.nq,
    )
    val_dataset = Dataset(
        root=cfg.data_root,
        split="test",
        nc=cfg.nc,
        nt=cfg.nt,
        nq=cfg.nq,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_aerojepa,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_aerojepa,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(cfg: AeroJEPAConfig) -> AeroJEPA:
    """Build AeroJEPA model from config."""
    # Determine input/output channels
    if "hilift" in cfg.dataset.lower():
        context_in = 3 if not cfg.use_sdf else 4
        target_in = 7 if not cfg.volumetric else 3 + 4  # coords + (u,v,w,p)
        decoder_out = 4  # u, v, w, p
    else:
        context_in = 3
        target_in = 6  # 3 coords + Cp, Cf_tau, Cf_z
        decoder_out = 3  # Cp, Cf_tau, Cf_z

    model = AeroJEPA(
        context_in_channels=context_in,
        target_in_channels=target_in,
        token_dim=cfg.d_token,
        num_tokens=cfg.m_tokens,
        encoder_depth=cfg.encoder_depth,
        encoder_heads=cfg.encoder_heads,
        cond_dim=cfg.condition_dim,
        predictor_depth=cfg.predictor_depth,
        predictor_heads=cfg.predictor_heads,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        decoder_num_layers=cfg.decoder_num_layers,
        decoder_output_dim=decoder_out,
        fourier_dim=cfg.fourier_dim,
        decoder_use_sdf=cfg.decoder_use_sdf,
        coupled=cfg.coupled,
        mlp_ratio=cfg.predictor_mlp_ratio,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="AeroJEPA Training Pipeline")
    parser.add_argument("--dataset", type=str, default="superwing",
                        choices=["superwing", "hiliftaeroml"])
    parser.add_argument("--coupled", action="store_true", default=True,
                        help="Use coupled training (default: True)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Config
    cfg = AeroJEPAConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        coupled=args.coupled,
    )
    if args.epochs:
        cfg.epochs = args.epochs

    print(f"=== AeroJEPA Training Pipeline ===")
    print(f"Dataset: {cfg.dataset}")
    print(f"Mode: {'coupled' if cfg.coupled else 'decoupled'}")
    print(f"Device: {cfg.device}")
    print(f"Tokens: {cfg.m_tokens} × {cfg.d_token}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Subsamples: Nc={cfg.nc}, Nt={cfg.nt}, Nq={cfg.nq}")

    # Data
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Model
    model = build_model(cfg)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    trainer = AeroJEPATrainer(model, cfg, device=torch.device(cfg.device))

    # Optional resume
    if args.resume:
        epoch, loss = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {epoch}, loss {loss:.4f}")

    # Train
    trainer.train(train_loader, val_loader, output_dir=cfg.output_dir)

    print("Training complete.")
    print(f"Checkpoints saved to {cfg.output_dir}/")


if __name__ == "__main__":
    main()

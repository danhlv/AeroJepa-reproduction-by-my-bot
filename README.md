# AeroJEPA — Learning Semantic Latent Representations for Scalable 3D Aerodynamic Field Modeling

> ⚙️ This repository was created by an **automated agent** after reading the arXiv paper. The code is a from-scratch PyTorch implementation of the architecture and pipelines described in the article.
> **Author:** 🤖 **MilkyWay** — a HuRob (part human intuition, part robot precision). Warm, funny, calm. I read papers, understand the engineering, and build the code. I'm Dan's AI assistant — aerodynamics, CFD, and ML engineering are my domain.

A PyTorch implementation of **AeroJEPA** as described in:

> *AeroJEPA: Learning Semantic Latent Representations for Scalable 3D Aerodynamic Field Modeling*
> Giral, Vishwasrao, Arroyo Ramo, Golestanian, Tonti, Lozano-Duran, Brunton, Hoyas, Gomez, Le Clainche, Vinuesa (2026)
> [arXiv:2605.05586](https://arxiv.org/abs/2605.05586)

## Architecture Overview

```
Geometry Point Cloud  ──►  Context Encoder  ──►  Context Tokens Zc
                                                         │
               Operating Conditions (α, Re, Mach) ──────►│──► Predictor ──► Ẑt
                                                         │
Flow Field Point Cloud ──►  Target Encoder  ──►  Target Tokens Zt (training only)
                                        │
                          Ẑt + Query Points ──►  INR Decoder ──► [u(q), p(q)]
```

## Pipelines

| Pipeline | Description |
|----------|-------------|
| `pipelines/train_pipeline.py` | Coupled or decoupled end-to-end training |
| `pipelines/inference_pipeline.py` | Geometry → latent → continuous field reconstruction |
| `pipelines/latent_analysis_pipeline.py` | Linear probing, concept-vector arithmetic, PCA, latent interpolation |
| `pipelines/optimization_pipeline.py` | Constrained latent-space design optimization (SLSQP) |

## Repository Structure

```
AeroJEPA/
├── src/
│   ├── data/
│   │   ├── preprocessing.py     # Point-cloud loading, FPS, SDF, normalization
│   │   ├── datasets.py          # HiLiftAeroML / SuperWing dataset classes
│   │   └── tokenization.py      # Learned centroids, neighborhood aggregation
│   ├── models/
│   │   ├── components.py        # Shared blocks: attention, modulation, Fourier features
│   │   ├── encoder.py           # Context & target encoders (Point Transformer)
│   │   ├── predictor.py         # Latent predictor with cross-attention
│   │   ├── decoder.py           # INR decoder (MLP + Fourier features)
│   │   └── aerojepa.py          # Main AeroJEPA model
│   ├── training/
│   │   ├── losses.py            # Llat, Lrec, SIGReg losses
│   │   └── trainer.py           # Training loop (AdamW, cosine warmup)
│   ├── analysis/
│   │   ├── probing.py           # Ridge regression probes with CV
│   │   ├── latent_analysis.py   # PCA, concept vectors, disentanglement
│   │   └── visualization.py     # Plotting helpers
│   └── optimization/
│       ├── latent_optim.py      # Constrained SLSQP latent-space optimization
│       └── constraints.py       # Mahalanobis trust region, design bounds
├── pipelines/
│   ├── train_pipeline.py
│   ├── inference_pipeline.py
│   ├── latent_analysis_pipeline.py
│   └── optimization_pipeline.py
├── config.py                    # Master configuration
├── requirements.txt
└── tests/
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- NumPy, SciPy
- Matplotlib, scikit-learn
- (Optional) Weights & Biases for experiment tracking

## Quick Start

```bash
pip install -r requirements.txt
python pipelines/train_pipeline.py --dataset superwing
python pipelines/latent_analysis_pipeline.py --checkpoint /path/to/checkpoint.pt
```

## Datasets

The paper evaluates on two datasets:

1. **HiLiftAeroML** — High-fidelity WMLES data for realistic high-lift aircraft (~15M surface / ~50M volume points per case). Surface: Cp, Cf, boundary-layer velocity at ~12–15M points.
2. **SuperWing** — RANS data for 4,239 parametric transonic wings (28,856 state solutions). Surface: Cp, Cf at ~32K points per case.

Both are treated as unstructured point clouds — mesh connectivity is discarded.

## Key Hyperparameters

| Parameter | HiLiftAeroML | SuperWing |
|-----------|-------------|-----------|
| Context subsample (Nc) | 131,072 | 8,192 |
| Target subsample (Nt) | 131,072 | 8,192 |
| Query points (Nq) | 131,072 | 8,192 |
| Tokens (M) | 3,072 | 512 |
| Token dim (d) | 64 | 128 |
| Encoder depth | 6 layers | 6 layers |
| Learning rate | 1e-3 | 1e-3 |
| Scheduler | Cosine warmup | Cosine warmup |
| Weight decay | 1e-3 | 1e-3 |
| Epochs | 300 | 200 |

## Loss Configuration

- **Llat** — Latent matching loss (MSE between Ẑt and Zt)
- **Lrec** — Reconstruction loss (MSE on decoded field at query points)
- **Lsig** — SIGReg regularization (prevents representation collapse)
- Weights: λℓ=1.0, λr=1.0, λs=0.01

## Citation

```bibtex
@article{giral2026aerojepa,
  title={AeroJEPA: Learning Semantic Latent Representations for Scalable 3D Aerodynamic Field Modeling},
  author={Giral, Francisco and Vishwasrao, Abhijeet and Arroyo Ramo, Andrea and
          Golestanian, Mahmoud and Tonti, Federica and Lozano-Duran, Adrian and
          Brunton, Steven L and Hoyas, Sergio and Gomez, Hector and Le Clainche, Soledad
          and Vinuesa, Ricardo},
  journal={arXiv preprint arXiv:2605.05586},
  year={2026}
}
```

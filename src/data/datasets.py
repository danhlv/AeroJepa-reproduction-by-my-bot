"""
Dataset classes for HiLiftAeroML and SuperWing  (§D.1, §D.2).

Both datasets are treated as unstructured point clouds — mesh connectivity is
discarded to match the AeroJEPA pipeline (§D).
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Callable


class AeroJEPADataset(Dataset):
    """Base dataset for AeroJEPA point-cloud data.

    Each sample must provide:
        - geometry_pts:  (Ng, 3)   boundary-coordinate point cloud
        - flow_pts:      (Nf, 3+C) flow-field point cloud (coords + values)
        - conditions:    (C_cond,) operating conditions (α, Re, Mach ...)
    Optional:
        - design_params: (D,)     ground-truth design parameter vector
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        nc: int = 8192,
        nt: int = 8192,
        nq: int = 8192,
        transform: Optional[Callable] = None,
        cache: bool = True,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.nc = nc
        self.nt = nt
        self.nq = nq
        self.transform = transform
        self.data = self._load()

    def _load(self) -> list:
        """Override in subclasses."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = dict(self.data[idx])
        # Subsample if needed — FPS is applied per sample
        from .preprocessing import farthest_point_sampling

        pc = sample["geometry_pts"]   # (N, 3)
        fc = sample["flow_pts"]       # (N_flow, 3 + C_flow)

        # Random subsample for training (paper §E.1 uses FPS; here we also allow random)
        if self.split == "train":
            perm = torch.randperm(pc.shape[0])[:self.nc]
            sample["geometry_pts"] = pc[perm]

            perm_f = torch.randperm(fc.shape[0])[:self.nt]
            sample["flow_pts"] = fc[perm_f]
        else:
            # Test: use FPS for deterministic evaluation
            pts, _ = farthest_point_sampling(pc.unsqueeze(0), self.nc)
            sample["geometry_pts"] = pts[0]
            pts_f, _ = farthest_point_sampling(fc[:, :3].unsqueeze(0), self.nt)
            sample["flow_pts"] = pts_f[0]

        if self.transform:
            sample = self.transform(sample)

        return sample


class HiLiftAeroMLDataset(AeroJEPADataset):
    """
    HiLiftAeroML: high-fidelity WMLES high-lift configurations.

    Surface: ~12–15M points per case with (x, y, z, u, v, w, p).
    8 continuous design parameters (control-surface deflection angles).
    10 AoA snapshots per geometry (4°–22°).

    Sizes (§D.1):
        Train: 205 geometry configurations
        Test:   50 geometry configurations

    Reference: Ashton et al., AIAA SCITECH 2026.
    """

    def __init__(self, root, split="train", nc=131072, nt=131072, nq=131072, transform=None):
        self.num_design_params = 8
        super().__init__(root, split, nc, nt, nq, transform)

    def _load(self):
        # Placeholder — replace with actual data loading logic:
        #   data_root / hiliftaeroml / train/  and  test/
        # Each folder contains .npz files with keys:
        #   surface_pts, surface_fields, design_params, conditions
        h = []
        split_dir = os.path.join(self.root, "hiliftaeroml", self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"HiLiftAeroML data not found at {split_dir}. "
                "See https://github.com/ashtonlab/HiLiftAeroML for download instructions."
            )
        for fname in sorted(os.listdir(split_dir)):
            if fname.endswith(".npz"):
                d = np.load(os.path.join(split_dir, fname))
                h.append({
                    "geometry_pts": torch.from_numpy(d["surface_pts"]).float(),
                    "flow_pts": torch.from_numpy(d["surface_fields"]).float(),
                    "conditions": torch.from_numpy(d["conditions"]).float(),
                    "design_params": torch.from_numpy(d["design_params"]).float(),
                })
        return h


class SuperWingDataset(AeroJEPADataset):
    """
    SuperWing: large-scale transonic swept-wing dataset.

    4,239 parametric wing geometries, 28,856 RANS state solutions.
    Surface: ~32K points per case with (Cp, Cf_tau, Cf_z).
    54 latent morphological design parameters.

    Split: 80/10/10 across geometries (not simulation sweeps) — zero-shot.

    Reference: Yang et al., arXiv:2512.14397, 2025.
    """

    def __init__(self, root, split="train", nc=8192, nt=8192, nq=8192, transform=None):
        self.num_design_params = 54
        super().__init__(root, split, nc, nt, nq, transform)

    def _load(self):
        h = []
        split_dir = os.path.join(self.root, "superwing", self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"SuperWing data not found at {split_dir}. "
                "See https://github.com/yangyunjia/SuperWing for download instructions."
            )
        for fname in sorted(os.listdir(split_dir)):
            if fname.endswith(".npz"):
                d = np.load(os.path.join(split_dir, fname))
                # Each file may contain multiple operating conditions per geometry
                surface_pts = torch.from_numpy(d["surface_pts"]).float()  # (N, 3)
                h.append({
                    "geometry_pts": surface_pts,
                    "flow_pts": torch.from_numpy(d["surface_fields"]).float(),
                    "conditions": torch.from_numpy(d["conditions"]).float(),
                    "design_params": torch.from_numpy(d["design_params"]).float(),
                })
        return h


def collate_aerojepa(batch: list) -> dict:
    """Collation function for AeroJEPA data batches."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == "design_params":
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            # Variable-size point clouds — keep as list
            out[k] = [b[k] for b in batch]
    return out

"""
Normalization utilities for ProtSCAPE datasets.
"""
import torch
import numpy as np

def normalize_energy(dataset):
    energies = np.array([float(getattr(g, "energy", 0.0)) for g in dataset], dtype=np.float64)
    mu = energies.mean() if energies.size else 0.0
    sd = energies.std() if energies.size else 1.0
    if sd == 0:
        sd = 1.0
    for g in dataset:
        g.energy = torch.tensor((float(getattr(g, "energy", 0.0)) - mu) / sd, dtype=torch.float32)
    return float(mu), float(sd)


def normalize_xyz_only(dataset, atom_type_dim: int, xyz_dim: int = 3):
    """
    Standardize ONLY the XYZ block in g.x:
      g.x[:, atom_type_dim:atom_type_dim+xyz_dim]
    across the entire dataset.
    """
    xyz_list = []
    for g in dataset:
        x = g.x.detach().cpu().numpy() if isinstance(g.x, torch.Tensor) else np.asarray(g.x)
        if x.shape[1] < atom_type_dim + xyz_dim:
            raise ValueError(
                f"normalize_xyz_only: x has F={x.shape[1]} but need atom_type_dim({atom_type_dim})+xyz_dim({xyz_dim})"
            )
        xyz_list.append(x[:, atom_type_dim:atom_type_dim + xyz_dim])

    X = np.concatenate(xyz_list, axis=0).astype(np.float64)  # (sum_nodes, 3)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0

    for g in dataset:
        x = g.x.detach().cpu().numpy() if isinstance(g.x, torch.Tensor) else np.asarray(g.x)
        xyz = x[:, atom_type_dim:atom_type_dim + xyz_dim]
        xyz_norm = (xyz - mu) / sd
        x_new = x.copy()
        x_new[:, atom_type_dim:atom_type_dim + xyz_dim] = xyz_norm
        g.x = torch.tensor(x_new, dtype=torch.float32)

    return mu.squeeze().astype(np.float32), sd.squeeze().astype(np.float32)

def replace_energy_with_linear_ramp(dataset, normalize=True):
    """
    Replace g.energy with a simple linear signal based on frame index.
    """
    N = len(dataset)
    energies = torch.linspace(0.0, 1.0, steps=N)

    if normalize:
        energies = (energies - energies.mean()) / energies.std()

    for i, g in enumerate(dataset):
        g.energy = energies[i].clone().detach()

    print("[energy] Replaced energies with linear ramp")

# -----------------------------
# Inference Normalization helpers
# -----------------------------

def compute_xyz_norm_stats(dataset, xyz_start: int = 3):
    """
    For minfeats layout: x = [Z, res_idx, aa_idx, xyz(3)]
    xyz slice is always [3:6] by default.
    """
    xyz_list = []
    for g in dataset:
        x = g.x.detach().cpu().numpy() if isinstance(g.x, torch.Tensor) else np.asarray(g.x)
        if x.shape[1] < xyz_start + 3:
            raise ValueError(f"x feature dim {x.shape[1]} too small for xyz slice [{xyz_start}:{xyz_start+3}]")
        xyz_list.append(x[:, xyz_start:xyz_start + 3])

    X = np.concatenate(xyz_list, axis=0).astype(np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_xyz_norm(dataset, mu, sd, xyz_start: int = 3):
    for g in dataset:
        x = g.x.detach().cpu().numpy() if isinstance(g.x, torch.Tensor) else np.asarray(g.x)
        x2 = x.copy()
        x2[:, xyz_start:xyz_start + 3] = (x2[:, xyz_start:xyz_start + 3] - mu) / sd
        g.x = torch.tensor(x2, dtype=torch.float32)
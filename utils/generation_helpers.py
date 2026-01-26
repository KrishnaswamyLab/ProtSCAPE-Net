"""
Helper functions for generation.py
Includes basic utilities, path handling, and data normalization.
"""

import os
import importlib.util
import pickle
from typing import Any, List, Optional, Tuple
import numpy as np
import torch


NM_TO_ANG = 10.0


def identity_denoiser(x: torch.Tensor) -> torch.Tensor:
    """Pass-through denoiser for latent space."""
    return x


def out_prefix_from_path(out_path: str) -> str:
    """Extract prefix from output path, removing .pkl or .pickle extensions."""
    base = out_path
    for ext in (".pkl", ".pickle"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return base


def ensure_tensor_latents(latents: Any) -> torch.Tensor:
    """Convert latent representations to torch.Tensor."""
    if isinstance(latents, torch.Tensor):
        return latents.float()
    if isinstance(latents, np.ndarray):
        return torch.from_numpy(latents).float()
    if isinstance(latents, list):
        pieces = []
        for l in latents:
            if isinstance(l, torch.Tensor):
                pieces.append(l.float())
            else:
                pieces.append(torch.tensor(l, dtype=torch.float))
        return torch.cat(pieces, dim=0)
    return torch.tensor(latents, dtype=torch.float)


def load_module_from_path(path: str, module_name: str = "atlas_new"):
    """Dynamically load a Python module from a file path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find module at: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def pick_default_pdb(protein: str) -> Optional[str]:
    """Find a PDB file for the given protein, checking common locations."""
    candidates = [
        f"{protein}_analysis/{protein}.pdb",
        "1v1h_C_analysis/1v1h_C.pdb",
        "1bxy_A_analysis/1bxy_A.pdb",
        "1bx7_A_analysis/1bx7_A.pdb",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def normalize_energy_inplace(dataset: List[Any]) -> Tuple[float, float, np.ndarray]:
    """
    Normalize energy values in the dataset using z-score normalization.
    
    Args:
        dataset: List of graph objects with 'energy' attribute
        
    Returns:
        Tuple of (mu, sd, raw_energy_array) where:
        - mu: Mean of raw energies
        - sd: Standard deviation of raw energies
        - raw_energy_array: Original energy values before normalization
    """
    raw = np.array([float(getattr(g, "energy", 0.0)) for g in dataset], dtype=np.float64)
    mu = float(raw.mean()) if raw.size else 0.0
    sd = float(raw.std()) if raw.size else 1.0
    if sd == 0.0:
        sd = 1.0
    for g in dataset:
        g.energy = torch.tensor((float(getattr(g, "energy", 0.0)) - mu) / sd, dtype=torch.float32)
    return mu, sd, raw


def get_energy_value(g: Any) -> float:
    """Extract energy value from a graph object as a float."""
    v = getattr(g, "energy", 0.0)
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().reshape(-1)[0].item())
    return float(v)


def safe_abs(path: str) -> str:
    """Convert a path to absolute path."""
    return os.path.abspath(path)


def load_mu_sd(args) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load normalization statistics (mean and std) for XYZ coordinates.
    
    Args:
        args: Argument namespace with xyz_stats_path, xyz_mu_path, and xyz_sd_path
        
    Returns:
        Tuple of (mu, sd) numpy arrays
    """
    if args.xyz_stats_path is not None:
        with open(args.xyz_stats_path, "rb") as f:
            d = pickle.load(f)
        mu = np.asarray(d["mu"], dtype=np.float64)
        sd = np.asarray(d["sd"], dtype=np.float64)
        return mu, sd

    if args.xyz_mu_path is None or args.xyz_sd_path is None:
        raise ValueError(
            "For --normalize_xyz you must provide either --xyz_stats_path "
            "or both --xyz_mu_path and --xyz_sd_path."
        )
    mu = np.load(args.xyz_mu_path).astype(np.float64)
    sd = np.load(args.xyz_sd_path).astype(np.float64)
    return mu, sd

"""
Visualization and PDB export functions for generation trajectories.
"""

import os
import subprocess
import shutil
import re
from typing import Any, Dict, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import phate
import MDAnalysis as mda

from utils.geometry import kabsch_align_np
from utils.generation_helpers import safe_abs


NM_TO_ANG = 10.0


def plot_multi_paths_pca(
    lat_all: np.ndarray,
    base_energy_norm: np.ndarray,
    traj: torch.Tensor,      # (T,B,D)
    out_path: str,
    title: str,
    plot_every: int = 1,
) -> None:
    """
    Plot multiple paths on PCA projection of latent space.
    
    Args:
        lat_all: Full dataset latent coordinates, shape (M, D)
        base_energy_norm: Normalized energies for color coding, shape (M,)
        traj: Trajectory in latent space, shape (T, B, D)
        out_path: Path to save figure
        title: Figure title
        plot_every: Plot every Nth timestep (for subsampling)
    """
    traj_np = traj.detach().cpu().numpy()
    T, B, D = traj_np.shape
    t_idx = np.arange(0, T, max(1, plot_every))
    traj_sub = traj_np[t_idx]  # (Tsub,B,D)

    p = PCA(n_components=2, random_state=0)
    base_2d = p.fit_transform(lat_all)

    plt.figure()
    plt.scatter(base_2d[:, 0], base_2d[:, 1], c=base_energy_norm, s=8, alpha=0.25)

    for b in range(B):
        path_2d = p.transform(traj_sub[:, b, :])
        plt.plot(path_2d[:, 0], path_2d[:, 1], linewidth=1.5, alpha=0.9)
        plt.scatter(path_2d[0, 0], path_2d[0, 1], marker="x", s=80, linewidths=2)
        plt.scatter(path_2d[-1, 0], path_2d[-1, 1], marker="x", s=80, linewidths=2)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_multi_paths_phate(
    lat_all: np.ndarray,
    base_energy_norm: np.ndarray,
    traj: torch.Tensor,      # (T,B,D)
    out_path: str,
    title: str,
    knn: int,
    t: int,
    plot_every: int = 1,
) -> None:
    """
    Plot multiple paths on PHATE projection of latent space.
    
    Args:
        lat_all: Full dataset latent coordinates, shape (M, D)
        base_energy_norm: Normalized energies for color coding, shape (M,)
        traj: Trajectory in latent space, shape (T, B, D)
        out_path: Path to save figure
        title: Figure title
        knn: Number of nearest neighbors for PHATE
        t: PHATE t parameter
        plot_every: Plot every Nth timestep (for subsampling)
    """
    traj_np = traj.detach().cpu().numpy()
    T, B, D = traj_np.shape
    t_idx = np.arange(0, T, max(1, plot_every))
    traj_sub = traj_np[t_idx]  # (Tsub,B,D)

    # PHATE on combined points
    combined = np.concatenate([lat_all, traj_sub.reshape(-1, D)], axis=0)
    ph = phate.PHATE(n_components=2, knn=knn, t=t, random_state=0)
    emb = ph.fit_transform(combined)
    base_2d = emb[: lat_all.shape[0]]
    traj_2d = emb[lat_all.shape[0] :].reshape(traj_sub.shape[0], B, 2)

    plt.figure()
    plt.scatter(base_2d[:, 0], base_2d[:, 1], c=base_energy_norm, s=8, alpha=0.25)

    for b in range(B):
        path_2d = traj_2d[:, b, :]
        plt.plot(path_2d[:, 0], path_2d[:, 1], linewidth=1.5, alpha=0.9)
        plt.scatter(path_2d[0, 0], path_2d[0, 1], marker="x", s=80, linewidths=2)
        plt.scatter(path_2d[-1, 0], path_2d[-1, 1], marker="x", s=80, linewidths=2)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def export_pdb_frames(
    xyz_nm: np.ndarray,                 # (T, N, 3)
    pdb_ref_path: str,
    out_dir: str,
    normalize_xyz: bool,
    xyz_mu: Optional[np.ndarray],
    xyz_sd: Optional[np.ndarray],
    n_pdb_frames: int,
    align_kabsch: bool,
    sel_atom_indices: Optional[np.ndarray] = None,
) -> str:
    """
    Export trajectory as PDB frames.
    
    Args:
        xyz_nm: Trajectory coordinates, shape (T, N, 3) in nanometers
        pdb_ref_path: Path to reference PDB file
        out_dir: Output directory for PDB files
        normalize_xyz: Whether coordinates were z-score normalized
        xyz_mu: Mean for denormalization
        xyz_sd: Std for denormalization
        n_pdb_frames: Maximum number of frames to export
        align_kabsch: Whether to align frames using Kabsch algorithm
        sel_atom_indices: Selection atom indices if xyz_nm is selection coords
        
    Returns:
        Absolute path to output directory
    """
    os.makedirs(out_dir, exist_ok=True)

    u = mda.Universe(pdb_ref_path)
    n_full = len(u.atoms)

    sel_idx = None
    if sel_atom_indices is not None:
        sel_idx = np.asarray(sel_atom_indices, dtype=int)

    full_ref_A = u.atoms.positions.astype(np.float64)

    def denorm_and_to_A(x_nm_frame: np.ndarray) -> np.ndarray:
        x = x_nm_frame.astype(np.float64)
        if normalize_xyz:
            assert xyz_mu is not None and xyz_sd is not None
            x = x * xyz_sd + xyz_mu
        return (x * NM_TO_ANG).astype(np.float64)

    def lift_to_full(pred_A_sel_or_full: np.ndarray) -> np.ndarray:
        if pred_A_sel_or_full.shape[0] == n_full:
            return pred_A_sel_or_full
        if sel_idx is None or pred_A_sel_or_full.shape[0] != len(sel_idx):
            raise ValueError(
                f"PDB has {n_full} atoms but decoded xyz has {pred_A_sel_or_full.shape[0]} atoms. "
                "Provide sel_atom_indices to lift selection coords to full, or decode full coords."
            )
        pred_A_full = full_ref_A.copy()
        pred_A_full[sel_idx] = pred_A_sel_or_full
        return pred_A_full

    ref_A = lift_to_full(denorm_and_to_A(xyz_nm[0]))

    for t in range(min(n_pdb_frames, xyz_nm.shape[0])):
        pred_A = denorm_and_to_A(xyz_nm[t])
        pred_A_full = lift_to_full(pred_A)

        if align_kabsch:
            try:
                pred_A_aligned, _ = kabsch_align_np(pred_A_full, ref_A)
            except Exception:
                pred_A_aligned = pred_A_full
        else:
            pred_A_aligned = pred_A_full

        u.atoms.positions = pred_A_aligned.astype(np.float32)
        u.atoms.write(os.path.join(out_dir, f"pred_frame_{t:05d}.pdb"))

    return safe_abs(out_dir)


def run_molprobity_on_folder(pdb_dir: str) -> Dict[str, float]:
    """
    Run MolProbity on all PDB files in a directory.
    
    Tries phenix.molprobity if available. If not found, returns empty dict.
    Parsing is best-effort; you may need to adjust regex for your environment output.
    
    Args:
        pdb_dir: Directory containing PDB files
        
    Returns:
        Dictionary with mean values of clashscore, rama_favored, rama_outliers, and n_frames_scored
    """
    phenix = shutil.which("phenix.molprobity")
    if phenix is None:
        print(f"[molprobity] phenix.molprobity not found; skipping for {pdb_dir}")
        return {}

    pdbs = sorted([os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith(".pdb")])
    if not pdbs:
        return {}

    clash = []
    rama_favored = []
    rama_outliers = []

    for pdb in pdbs:
        try:
            out = subprocess.check_output([phenix, pdb], stderr=subprocess.STDOUT, text=True)

            m1 = re.search(r"Clashscore\s*=\s*([0-9.]+)", out)
            if m1:
                clash.append(float(m1.group(1)))

            m2 = re.search(r"Ramachandran favored\s*=\s*([0-9.]+)%", out)
            if m2:
                rama_favored.append(float(m2.group(1)))

            m3 = re.search(r"Ramachandran outliers\s*=\s*([0-9.]+)%", out)
            if m3:
                rama_outliers.append(float(m3.group(1)))

        except subprocess.CalledProcessError as e:
            msg = e.output if isinstance(e.output, str) else str(e)
            print(f"[molprobity warn] failed on {pdb}: {msg[:400]}")

    def agg(x):
        return float(np.mean(x)) if len(x) else float("nan")

    return {
        "clashscore_mean": agg(clash),
        "rama_favored_mean": agg(rama_favored),
        "rama_outliers_mean": agg(rama_outliers),
        "n_frames_scored": float(len(pdbs)),
    }

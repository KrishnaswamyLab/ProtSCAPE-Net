"""
Visualization and PDB export functions for generation trajectories.
"""

import os
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
        plt.scatter(path_2d[-1, 0], path_2d[-1, 1], marker="o", s=80, linewidths=2)

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
        plt.scatter(path_2d[-1, 0], path_2d[-1, 1], marker="o", s=80, linewidths=2)

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


def compute_structure_metrics(pdb_dir: str) -> Optional[Dict[str, float]]:
    """
    Compute structure quality metrics manually without external tools.
    
    Computes:
    - Clashscore: Number of serious steric clashes per 1000 atoms
    - Ramachandran statistics: Percentage of residues in favored/outlier regions
    
    Args:
        pdb_dir: Directory containing PDB files
        
    Returns:
        Dictionary with mean values of clashscore, rama_favored, rama_outliers, and n_frames_scored
    """
    pdbs = sorted([os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith(".pdb")])
    # pdbs = ["Inference/7jfl/pdb_frames/true_frame_00005.pdb"]
    if not pdbs:
        return {}

    clash_scores = []
    rama_favored = []
    rama_outliers = []

    for pdb_path in pdbs:
        try:
            u = mda.Universe(pdb_path)
            
            # Compute clashscore (atoms too close together)
            n_clashes = compute_clashes(u)
            n_atoms = len(u.atoms)
            clashscore = (n_clashes / n_atoms) * 1000 if n_atoms > 0 else 0.0
            clash_scores.append(clashscore)
            
            # Compute Ramachandran statistics
            rama_stats = compute_ramachandran_stats(u)
            if rama_stats is not None:
                rama_favored.append(rama_stats['favored_pct'])
                rama_outliers.append(rama_stats['outlier_pct'])
                
        except Exception as e:
            print(f"[metrics warn] failed on {pdb_path}: {str(e)[:200]}")

    def agg(x):
        return float(np.mean(x)) if len(x) else float("nan")

    def median(x):
        return float(np.median(x)) if len(x) else float("nan")

    return {
        "clashscore_mean": agg(clash_scores),
        "rama_favored_mean": agg(rama_favored),
        "rama_outliers_mean": agg(rama_outliers),
        "clashscore_median": median(clash_scores),
        "rama_favored_median": median(rama_favored),
        "rama_outliers_median": median(rama_outliers),
        "n_frames_scored": float(len(pdbs)),
    }


def compute_clashes(u: mda.Universe, clash_distance: float = 2.0) -> int:
    """
    Count steric clashes (non-bonded atoms too close together).
    
    Args:
        u: MDAnalysis Universe
        clash_distance: Distance threshold in Angstroms (default: 2.0)
        
    Returns:
        Number of clashes detected
    """
    # Get all heavy atoms (non-hydrogen)
    heavy = u.select_atoms("not name H*")
    positions = heavy.positions
    
    n_clashes = 0
    # Check pairwise distances
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            # Check if atoms are close but not bonded
            # Simple heuristic: if distance < clash_distance, it's a clash
            if dist < clash_distance:
                # Check if bonded (simple connectivity check)
                atom_i = heavy[i]
                atom_j = heavy[j]
                # Skip if same residue (likely bonded)
                if atom_i.resid != atom_j.resid:
                    # Skip if adjacent residues and backbone atoms (peptide bond)
                    if abs(atom_i.resid - atom_j.resid) == 1:
                        if atom_i.name in ['C', 'N', 'CA', 'O'] and atom_j.name in ['C', 'N', 'CA', 'O']:
                            continue
                    n_clashes += 1
    
    return n_clashes


def compute_ramachandran_stats(u: mda.Universe) -> Optional[Dict[str, float]]:
    """
    Compute Ramachandran statistics (phi/psi backbone dihedral angles).
    
    Args:
        u: MDAnalysis Universe
        
    Returns:
        Dictionary with favored_pct and outlier_pct, or None if no protein
    """
    try:
        protein = u.select_atoms("protein")
        if len(protein) == 0:
            return None
        
        # Get all residues
        residues = protein.residues
        
        phi_psi_pairs = []
        for res in residues:
            # Need consecutive residues for phi/psi calculation
            try:
                # Phi: C(i-1) - N(i) - CA(i) - C(i)
                # Psi: N(i) - CA(i) - C(i) - N(i+1)
                
                # Get atoms
                if 'CA' not in [a.name for a in res.atoms]:
                    continue
                    
                ca = res.atoms.select_atoms("name CA")[0]
                n = res.atoms.select_atoms("name N")
                c = res.atoms.select_atoms("name C")
                
                if len(n) == 0 or len(c) == 0:
                    continue
                    
                n = n[0]
                c = c[0]
                
                # Calculate phi (needs previous residue)
                phi = None
                if res.resid > residues[0].resid:
                    prev_res = residues[res.resid - residues[0].resid - 1]
                    prev_c = prev_res.atoms.select_atoms("name C")
                    if len(prev_c) > 0:
                        phi = compute_dihedral(prev_c[0].position, n.position, 
                                              ca.position, c.position)
                
                # Calculate psi (needs next residue)
                psi = None
                if res.resid < residues[-1].resid:
                    next_res = residues[res.resid - residues[0].resid + 1]
                    next_n = next_res.atoms.select_atoms("name N")
                    if len(next_n) > 0:
                        psi = compute_dihedral(n.position, ca.position,
                                              c.position, next_n[0].position)
                
                if phi is not None and psi is not None:
                    phi_psi_pairs.append((phi, psi))
                    
            except Exception:
                continue
        
        if len(phi_psi_pairs) == 0:
            return None
        
        # Classify using Ramachandran regions
        n_favored = 0
        n_outliers = 0
        
        for phi, psi in phi_psi_pairs:
            if is_ramachandran_favored(phi, psi):
                n_favored += 1
            elif is_ramachandran_outlier(phi, psi):
                n_outliers += 1
        
        total = len(phi_psi_pairs)
        return {
            'favored_pct': (n_favored / total) * 100.0,
            'outlier_pct': (n_outliers / total) * 100.0,
        }
        
    except Exception as e:
        print(f"[ramachandran warn] {str(e)[:200]}")
        return None


def compute_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Compute dihedral angle in degrees for four points.
    
    Args:
        p1, p2, p3, p4: 3D coordinates of four atoms
        
    Returns:
        Dihedral angle in degrees
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    return np.degrees(np.arctan2(y, x))


def is_ramachandran_favored(phi: float, psi: float) -> bool:
    """
    Check if phi/psi angles are in favored Ramachandran regions.
    
    Simplified regions based on standard Ramachandran plot:
    - Alpha helix: phi ~ -60, psi ~ -45
    - Beta sheet: phi ~ -120, psi ~ +120
    
    Args:
        phi, psi: Backbone dihedral angles in degrees
        
    Returns:
        True if in favored region
    """
    # Alpha helix region
    if -100 <= phi <= -30 and -70 <= psi <= -10:
        return True
    # Beta sheet region
    if -180 <= phi <= -90 and 90 <= psi <= 180:
        return True
    # Left-handed alpha helix
    if 30 <= phi <= 90 and -10 <= psi <= 50:
        return True
    return False


def is_ramachandran_outlier(phi: float, psi: float) -> bool:
    """
    Check if phi/psi angles are in outlier Ramachandran regions.
    
    Args:
        phi, psi: Backbone dihedral angles in degrees
        
    Returns:
        True if in outlier region
    """
    # Simplified: outliers are typically in regions not favored or allowed
    # This is a conservative definition
    if is_ramachandran_favored(phi, psi):
        return False
    
    # Allowed regions (less favorable but not outliers)
    # Extended regions around favored areas
    if -180 <= phi <= -30 and -90 <= psi <= 50:
        return False
    if -180 <= phi <= 0 and 90 <= psi <= 180:
        return False
    if 0 <= phi <= 90 and -60 <= psi <= 90:
        return False
    
    # Everything else is an outlier
    return True

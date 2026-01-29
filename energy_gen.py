"""
NEB-based conformational path generation with comprehensive validation.

This script:
1. Loads a trajectory dataset and uses the FIRST and LAST frames as endpoints
2. Generates paths between start and end using Nudged Elastic Band (NEB)
3. Decodes paths to Cartesian coordinates and exports PDBs
4. Computes comprehensive structural metrics (MolProbity + inter-frame RMSD)

Usage:
  python energy_gen.py --config configs/config_generation_neb.yaml

The script will automatically use:
  - Start point: First frame in the trajectory (index 0)
  - End point: Last frame in the trajectory (index -1)
"""

from __future__ import annotations

import glob
import os
import pickle
import time
import re
import subprocess
from typing import Callable, Optional, Any, List, Tuple, Dict
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from protscape.protscape import ProtSCAPE
from protscape.neb import generate_neb_paths

# Import helper functions
from utils.config import load_config, config_to_hparams
from utils.generation_helpers import (
    ensure_tensor_latents,
    load_module_from_path,
    pick_default_pdb,
    normalize_energy_inplace,
    get_energy_value,
)
from utils.openmm_eval import build_grad_fn_openmm
from utils.geometry import kabsch_align_np

NM_TO_ANG = 10.0

NM_TO_ANG = 10.0

# ============================================================================
# MolProbity Functions (from ensemble_gen.py)
# ============================================================================

def check_phenix_available() -> bool:
    """Check if phenix.molprobity is available."""
    try:
        result = subprocess.run(
            ['which', 'phenix.molprobity'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def run_phenix_molprobity(pdb_path: str):
    """
    Run phenix.molprobity on a PDB file.
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        Dictionary with MolProbity scores, or None if execution fails
    """
    try:
        # Convert to absolute path
        abs_pdb_path = os.path.abspath(pdb_path)
        
        # Run phenix.molprobity with keep_hydrogens to avoid adding/removing H
        result = subprocess.run(
            ['phenix.molprobity', abs_pdb_path, 'keep_hydrogens=True'],
            capture_output=True,
            text=True,
            timeout=120
        )
        # Parse output (check both stdout and stderr)
        output = result.stdout + result.stderr
        scores = {}
        
        # Extract Ramachandran outliers (various formats)
        patterns = [
            r'Ramachandran outliers\s*[=:]\s*([\d.]+)\s*%',
            r'ramachandran_outliers\s*[=:]\s*([\d.]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                scores['ramachandran_outliers'] = float(match.group(1))
                break
        
        # Extract Clashscore
        patterns = [
            r'Clashscore\s*[=:]\s*([\d.]+)',
            r'clashscore\s*[=:]\s*([\d.]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                scores['clashscore'] = float(match.group(1))
                break
        
        # Extract Rotamer outliers
        patterns = [
            r'Rotamer outliers\s*[=:]\s*([\d.]+)\s*%',
            r'rotamer_outliers\s*[=:]\s*([\d.]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                scores['rotamer_outliers'] = float(match.group(1))
                break
        
        # Extract MolProbity score
        patterns = [
            r'MolProbity score\s*[=:]\s*([\d.]+)',
            r'molprobity_score\s*[=:]\s*([\d.]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                scores['molprobity_score'] = float(match.group(1))
                break
        
        # Return scores if we got at least some metrics
        if len(scores) >= 2:  # At least 2 metrics extracted
            return scores
        else:
            return None
        
    except Exception:
        return None


def compute_structure_metrics_all_frames(pdb_dir: str):
    """
    Compute comprehensive MolProbity scores for all PDB frames.
    
    Args:
        pdb_dir: Directory containing PDB files
        
    Returns:
        List of dictionaries with metrics for each frame
    """
    pdbs = sorted([f for f in os.listdir(pdb_dir) if f.endswith(".pdb")])
    if not pdbs:
        print("[warn] No PDB files found")
        return None
    
    # Check if phenix.molprobity is available
    phenix_available = check_phenix_available()
    if not phenix_available:
        print("[error] phenix.molprobity not found. Please install PHENIX or load the module.")
        return None
    
    print("[molprobity] Using phenix.molprobity for scoring")
    all_metrics = []
    
    for i, pdb_file in enumerate(pdbs):
        pdb_path = os.path.join(pdb_dir, pdb_file)
        
        try:
            # Use phenix.molprobity
            result = run_phenix_molprobity(pdb_path)
            
            if result:
                metrics = {
                    'frame': i,
                    'pdb_file': pdb_file,
                    'rama_outliers_pct': result.get('ramachandran_outliers', np.nan),
                    'clashscore': result.get('clashscore', np.nan),
                    'rotamer_outliers_pct': result.get('rotamer_outliers', np.nan),
                    'molprobity_score': result.get('molprobity_score', np.nan),
                }
            else:
                # Phenix failed for this file
                print(f"[warn] phenix.molprobity failed for {pdb_file}")
                metrics = {
                    'frame': i,
                    'pdb_file': pdb_file,
                    'rama_outliers_pct': np.nan,
                    'clashscore': np.nan,
                    'rotamer_outliers_pct': np.nan,
                    'molprobity_score': np.nan,
                }
            
            all_metrics.append(metrics)
            
            if (i + 1) % 10 == 0:
                print(f"[progress] Scored {i + 1}/{len(pdbs)} frames")
                
        except Exception as e:
            print(f"[warn] Failed to score {pdb_file}: {str(e)[:100]}")
            all_metrics.append({
                'frame': i,
                'pdb_file': pdb_file,
                'rama_outliers_pct': np.nan,
                'clashscore': np.nan,
                'rotamer_outliers_pct': np.nan,
                'molprobity_score': np.nan,
            })
    
    return all_metrics


def compute_interframe_rmsd(xyz_path_A: np.ndarray) -> np.ndarray:
    """
    Compute RMSD in Angstroms between consecutive frames.
    
    Args:
        xyz_path_A: Coordinates in Angstroms, shape (T, N, 3)
        
    Returns:
        Array of RMSDs, shape (T-1,)
    """
    T = xyz_path_A.shape[0]
    if T < 2:
        return np.array([])
    
    rmsds = []
    for i in range(T - 1):
        xyz1 = xyz_path_A[i]
        xyz2 = xyz_path_A[i + 1]
        diff = xyz2 - xyz1
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))
        rmsds.append(rmsd)
    
    return np.array(rmsds)


def compute_pairwise_rmsd(xyz1: np.ndarray, xyz2: np.ndarray) -> float:
    """
    Compute RMSD in Angstroms between two structures.
    
    Args:
        xyz1: Coordinates in Angstroms, shape (N, 3)
        xyz2: Coordinates in Angstroms, shape (N, 3)
        
    Returns:
        RMSD in Angstroms
    """
    diff = xyz2 - xyz1
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))
    return rmsd


def find_max_rmsd_pair(latents_all: np.ndarray, model, device: torch.device, 
                       max_frames: int = 500, u=None, ag=None, ref_xyz_A=None, 
                       out_dir=None) -> Tuple[int, int, float]:
    """
    Find the pair of frames with maximum RMSD.
    
    Args:
        latents_all: All latent representations, shape (T, D)
        model: The ProtSCAPE model for decoding latents
        device: Torch device
        max_frames: Maximum number of frames to consider (for computational efficiency)
        u: MDAnalysis Universe (optional, for PDB export)
        ag: MDAnalysis AtomGroup (optional, for PDB export)
        ref_xyz_A: Reference coordinates in Angstroms (optional, for alignment)
        out_dir: Output directory for PDB files (optional)
        
    Returns:
        Tuple of (start_idx, end_idx, max_rmsd)
    """
    T = len(latents_all)
    
    # If too many frames, sample uniformly
    if T > max_frames:
        print(f"[rmsd] Sampling {max_frames} frames from {T} total for RMSD computation")
        indices = np.linspace(0, T-1, max_frames, dtype=int)
    else:
        indices = np.arange(T)
    
    print(f"[rmsd] Decoding {len(indices)} frames to compute pairwise RMSDs...")
    
    # Decode all sampled frames
    xyz_frames = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_latents = latents_all[batch_indices]
            batch_latents_tensor = torch.tensor(batch_latents, dtype=torch.float32, device=device)
            
            xyz_batch = model.reconstruct_xyz(batch_latents_tensor)
            xyz_np = xyz_batch.detach().cpu().numpy()
            
            # Ensure shape is (batch, N, 3)
            if xyz_np.ndim == 3 and xyz_np.shape[-1] == 3:
                xyz_frames.append(xyz_np)
            else:
                xyz_reshaped = xyz_np.reshape(xyz_np.shape[0], -1, 3)
                xyz_frames.append(xyz_reshaped)
    
    xyz_all = np.concatenate(xyz_frames, axis=0)  # (n_samples, N, 3) in nm
    xyz_all_A = xyz_all * NM_TO_ANG  # Convert to Angstroms
    
    print(f"[rmsd] Computing pairwise RMSDs for {len(indices)} frames...")
    
    # Compute pairwise RMSDs
    max_rmsd = 0.0
    best_i = 0
    best_j = 1
    
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            rmsd = compute_pairwise_rmsd(xyz_all_A[i], xyz_all_A[j])
            if rmsd > max_rmsd:
                max_rmsd = rmsd
                best_i = i
                best_j = j
        
        if (i + 1) % 50 == 0:
            print(f"[progress] Processed {i+1}/{len(indices)} frames...")
    
    # Map back to original indices
    start_idx = indices[best_i]
    end_idx = indices[best_j]
    
    print(f"\n[rmsd] Maximum RMSD found: {max_rmsd:.3f} Å")
    print(f"[rmsd] Between frame {start_idx} and frame {end_idx}")
    
    # Export start and end structures as PDBs if parameters provided
    if u is not None and ag is not None and ref_xyz_A is not None and out_dir is not None:
        print(f"[rmsd] Exporting start and end structures as PDBs...")
        
        # Decode the actual start and end frames (not just sampled ones)
        with torch.no_grad():
            start_latent = torch.tensor(latents_all[start_idx:start_idx+1], dtype=torch.float32, device=device)
            end_latent = torch.tensor(latents_all[end_idx:end_idx+1], dtype=torch.float32, device=device)
            
            start_xyz = model.reconstruct_xyz(start_latent).detach().cpu().numpy()
            end_xyz = model.reconstruct_xyz(end_latent).detach().cpu().numpy()
            
            # Ensure correct shape
            if start_xyz.ndim == 3:
                start_xyz = start_xyz[0]  # (N, 3)
            else:
                start_xyz = start_xyz.reshape(-1, 3)
            
            if end_xyz.ndim == 3:
                end_xyz = end_xyz[0]  # (N, 3)
            else:
                end_xyz = end_xyz.reshape(-1, 3)
        
        # Convert to Angstroms
        start_xyz_A = start_xyz * NM_TO_ANG
        end_xyz_A = end_xyz * NM_TO_ANG
        
        # Kabsch align to reference
        try:
            start_xyz_aligned_A, _ = kabsch_align_np(start_xyz_A, ref_xyz_A)
            end_xyz_aligned_A, _ = kabsch_align_np(end_xyz_A, ref_xyz_A)
        except Exception as e:
            print(f"[warn] Kabsch alignment failed: {e}")
            start_xyz_aligned_A = start_xyz_A
            end_xyz_aligned_A = end_xyz_A
        
        # Export PDBs
        start_pdb_path = out_dir / f"start_frame_{start_idx:05d}.pdb"
        end_pdb_path = out_dir / f"end_frame_{end_idx:05d}.pdb"
        
        ag.positions = start_xyz_aligned_A.astype(np.float32)
        ag.write(str(start_pdb_path))
        
        ag.positions = end_xyz_aligned_A.astype(np.float32)
        ag.write(str(end_pdb_path))
        
        print(f"[saved] Start structure: {start_pdb_path}")
        print(f"[saved] End structure: {end_pdb_path}")
    
    return start_idx, end_idx, max_rmsd


def visualize_latent_space_with_path(
    latents_all: np.ndarray,
    path_latents: np.ndarray,
    full_dataset: list,
    start_idx: int,
    end_idx: int,
    out_path: str,
    protein_name: str = "",
    path_idx: int = 0
):
    """
    Visualize the latent space with PCA and overlay the NEB path.
    
    Args:
        latents_all: All latent representations, shape (T, D)
        path_latents: NEB path latent representations, shape (T_path, D)
        full_dataset: Dataset with energy information
        start_idx: Index of start frame
        end_idx: Index of end frame
        out_path: Path to save the figure
        protein_name: Name of protein for title
        path_idx: Index of path being visualized
    """
    print(f"\n[viz] Creating latent space visualization...")
    
    # Extract energies from dataset
    energies = np.array([get_energy_value(g) for g in full_dataset])
    
    # Perform PCA
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents_all)
    path_pca = pca.transform(path_latents)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all trajectory points colored by energy
    scatter = ax.scatter(
        latents_pca[:, 0],
        latents_pca[:, 1],
        c=energies,
        cmap='viridis',
        alpha=0.6,
        s=20,
        edgecolors='none'
    )
    
    # Add colorbar for energy
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Energy (kcal/mol)', fontsize=12)
    
    # Plot the NEB path
    ax.plot(
        path_pca[:, 0],
        path_pca[:, 1],
        'r-',
        linewidth=2,
        alpha=0.8,
        label=f'NEB Path {path_idx}',
        zorder=10
    )
    
    # Mark start point
    ax.scatter(
        latents_pca[start_idx, 0],
        latents_pca[start_idx, 1],
        c='orange',
        s=300,
        marker='*',
        edgecolors='black',
        linewidths=2,
        label=f'Start (frame {start_idx})',
        zorder=15
    )
    
    # Mark end point
    ax.scatter(
        latents_pca[end_idx, 0],
        latents_pca[end_idx, 1],
        c='red',
        s=300,
        marker='*',
        edgecolors='black',
        linewidths=2,
        label=f'End (frame {end_idx})',
        zorder=15
    )
    
    # Labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
    
    title = f'PCA multipath (NEB): {protein_name} (best={path_idx})'
    ax.set_title(title, fontsize=16)
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[saved] Latent space visualization: {out_path}")
    print(f"[viz] PCA explains {pca.explained_variance_ratio_[:2].sum()*100:.1f}% of variance")


# ============================================================================
# Main execution
# ============================================================================

def main():
    """
    Main NEB generation script.
    
    Usage:
      python energy_gen.py --config configs/config_generation_neb.yaml
    """
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config file (.yaml or .json)")
    
    # Allow command-line overrides
    cli_args, remaining = parser.parse_known_args()

    # Load config from file
    config = load_config(cli_args.config)
    
    # Convert to hparams namespace
    args = config_to_hparams(config)
    
    # Parse remaining args as overrides
    override_parser = ArgumentParser()
    for key, value in config.items():
        if isinstance(value, bool):
            override_parser.add_argument(f"--{key}", action="store_true")
        elif isinstance(value, int):
            override_parser.add_argument(f"--{key}", type=int)
        elif isinstance(value, float):
            override_parser.add_argument(f"--{key}", type=float)
        else:
            override_parser.add_argument(f"--{key}", type=str)
    
    override_args = override_parser.parse_args(remaining)
    
    # Apply overrides
    for key, value in vars(override_args).items():
        if value is not None:
            setattr(args, key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ========================================================================
    # Dataset Load
    # ========================================================================
    protein_id = args.protein.lower()
    dataset_path = 'data/graphs/'
    
    # Try pattern: {protein_id}_{chain}_graphs.pkl
    pattern = f"{protein_id}_A_graphs.pkl"
    matches = glob.glob(os.path.join(dataset_path, pattern))
    
    if matches:
        dataset_path = matches[0]
        print(f"[dataset] auto-detected: {dataset_path}")
    else:
        raise FileNotFoundError(
            f"No dataset found matching pattern '{pattern}'. "
            f"Expected format: {{pdbid}}_{{chain}}_graphs.pkl"
        )

    with open(dataset_path, "rb") as f:
        full_dataset = pickle.load(f)
    if len(full_dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    # Normalize dataset energies
    energy_mu, energy_sd, base_energy_raw = normalize_energy_inplace(full_dataset)
    print(f"[energy] normalized dataset energies with mu={energy_mu:.6f}, sd={energy_sd:.6f}")

    # ========================================================================
    # Model Setup
    # ========================================================================
    args.num_nodes = full_dataset[0].x.shape[0]
    args.node_feat_dim = full_dataset[0].x.shape[1]
    args.input_dim = 3
    args.prot_graph_size = args.num_nodes

    Z_max = int(max(g.x[:, 0].max().item() for g in full_dataset))
    res_max = int(max(g.x[:, 1].max().item() for g in full_dataset))
    aa_max = int(max(g.x[:, 2].max().item() for g in full_dataset))
    print("[vocab]", Z_max, res_max, aa_max)

    args.num_Z = Z_max + 1
    args.num_residues = res_max + 1
    args.num_aa = max(aa_max + 1, 21)

    # Latents
    latents_all = np.load(args.latent_path, allow_pickle=True)
    latents_all = ensure_tensor_latents(latents_all)
    print(f"[latents] shape={tuple(latents_all.shape)}")

    # Model load
    model = ProtSCAPE(args).to(device).eval()
    weights = torch.load(args.model_path, map_location=device)
    model.load_state_dict(weights)
    print(f"[model] Loaded from {args.model_path}")

    # ========================================================================
    # OpenMM Setup
    # ========================================================================
    atlas_rt = load_module_from_path(args.atlas_rt_path, module_name="atlas_new")
    if not hasattr(atlas_rt, "setup_simulation"):
        raise AttributeError(f"{args.atlas_rt_path} does not define setup_simulation")
    setup_simulation = getattr(atlas_rt, "setup_simulation")

    pdb_path = args.pdb or pick_default_pdb(args.protein)
    if pdb_path is None or not os.path.exists(pdb_path):
        raise FileNotFoundError("No PDB found. Provide --pdb or place a PDB in a known analysis folder.")
    print(f"[pdb] {pdb_path}")

    # Build latent gradient fn
    grad_fn = build_grad_fn_openmm(
        model=model,
        dataset=full_dataset,
        setup_simulation_fn=setup_simulation,
        pdb_path=pdb_path,
        device=device,
    )

    # ========================================================================
    # Setup PDB Export (need this before finding max RMSD pair)
    # ========================================================================
    u = mda.Universe(pdb_path)
    
    sel_idx = None
    if hasattr(full_dataset[0], "sel_atom_indices"):
        sel_idx = full_dataset[0].sel_atom_indices.detach().cpu().numpy().astype(int)
        print(f"[decode] Using selection indices: {len(sel_idx)} atoms")
    
    if sel_idx is None:
        raise RuntimeError(
            "Dataset graphs do not have sel_atom_indices. "
            "Please store sel_atom_indices during preprocessing so export ordering matches."
        )
    
    ag = u.atoms[sel_idx]
    print(f"[info] Using stored sel_atom_indices for export: {len(ag)} atoms")
    
    if len(ag) != args.num_nodes:
        raise ValueError(
            f"MDAnalysis AtomGroup has {len(ag)} atoms but graphs have {args.num_nodes} nodes. "
            "Counts must match for PDB export."
        )
    
    # Get reference structure in Å for alignment
    ref_xyz_A = ag.positions.astype(np.float64)
    
    # Create output directory
    out_dir = Path(args.out).parent
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # ========================================================================
    # Define Start and End Points (MAXIMUM RMSD pair)
    # ========================================================================
    print("\n" + "="*80)
    print("Finding start and end frames with maximum RMSD...")
    print("="*80)
    
    start_idx, end_idx, max_rmsd = find_max_rmsd_pair(
        latents_all=latents_all,
        model=model,
        device=device,
        max_frames=500,  # Adjust if needed for computational efficiency
        u=u,
        ag=ag,
        ref_xyz_A=ref_xyz_A,
        out_dir=out_dir
    )
    
    print(f"\n[endpoints] Selected frames with maximum RMSD:")
    print(f"  Start: frame {start_idx}")
    print(f"  End:   frame {end_idx}")
    print(f"  RMSD:  {max_rmsd:.3f} Å")
    print("="*80)
    
    # For n_paths, we'll generate that many independent NEB paths
    # all going from start to end (or we can use different start points)
    n_paths = getattr(args, 'n_paths', 1)
    
    # Use the same start and end for all paths (or modify if needed)
    start_indices = [start_idx for _ in range(n_paths)]
    
    # Set end_idx in args so NEB uses it
    args.end_idx = end_idx
    
    # ========================================================================
    # Generate NEB Paths
    # ========================================================================
    print(f"\n[NEB] Generating {n_paths} path(s) from frame {start_idx} to frame {end_idx}")
    print(f"[NEB] Parameters: n_pivots={getattr(args, 'n_pivots', 20)}, neb_steps={getattr(args, 'neb_steps', 50)}, neb_lr={getattr(args, 'neb_lr', 0.05)}")
    
    t0 = time.time()
    traj = generate_neb_paths(
        grad_fn=grad_fn,
        latents_all=latents_all,
        start_indices=start_indices,
        args=args,
        device=device,
        model=model,
    )
    t1 = time.time()
    print(f"[done] NEB generation time={t1 - t0:.2f}s; traj shape={tuple(traj.shape)}")  # (T,B,D)

    # ========================================================================
    # Decode Paths to XYZ
    # ========================================================================
    print("\n[decode] Decoding paths to xyz coordinates...")
    
    # Decode all frames
    T, B, D = traj.shape
    xyz_nm_list = []
    batch_size = 16
    with torch.no_grad():
        for b in range(B):
            path_latents = traj[:, b, :]  # (T, D)
            xyz_path = []
            for i in range(0, T, batch_size):
                batch = path_latents[i:i+batch_size]
                xyz_batch = model.reconstruct_xyz(batch)
                xyz_np = xyz_batch.detach().cpu().numpy()
                
                # Ensure shape is (batch, N, 3)
                if xyz_np.ndim == 3 and xyz_np.shape[-1] == 3:
                    xyz_path.append(xyz_np)
                else:
                    xyz_reshaped = xyz_np.reshape(xyz_np.shape[0], -1, 3)
                    xyz_path.append(xyz_reshaped)
            
            xyz_path_nm = np.concatenate(xyz_path, axis=0)  # (T, N, 3)
            xyz_nm_list.append(xyz_path_nm)
    
    xyz_paths_nm = np.stack(xyz_nm_list, axis=1)  # (T, B, N, 3)
    print(f"[decode] xyz_paths_nm shape={xyz_paths_nm.shape} (T, B, N, 3) in nm")
    
    # ========================================================================
    # Denormalize if needed
    # ========================================================================
    if args.normalize_xyz:
        if args.xyz_mu_path and args.xyz_sd_path:
            xyz_mu = np.load(args.xyz_mu_path)
            xyz_sd = np.load(args.xyz_sd_path)
            print(f"[decode] Loaded normalization params from {args.xyz_mu_path} and {args.xyz_sd_path}")
            # Denormalize: xyz = xyz * sd + mu (nm)
            xyz_paths_nm = xyz_paths_nm * xyz_sd + xyz_mu
            print("[decode] Applied denormalization: xyz = xyz*sd + mu")
        else:
            print("[warn] normalize_xyz=True but no xyz_mu_path/xyz_sd_path provided, skipping denormalization")
    
    # ========================================================================
    # Export PDBs and Compute Metrics (per path)
    # ========================================================================
    
    prefix = str(Path(args.out).stem)
    base_pdb_dir = out_dir / f"{prefix}_neb_paths"
    os.makedirs(base_pdb_dir, exist_ok=True)
    
    all_metrics = []
    
    for b in range(B):
        print(f"\n[path {b+1}/{B}] Exporting PDBs and computing metrics...")
        
        # Create path-specific directory
        pdb_dir_b = base_pdb_dir / f"path_{b:02d}"
        os.makedirs(pdb_dir_b, exist_ok=True)
        
        # Export all frames for this path
        xyz_path_nm = xyz_paths_nm[:, b, :, :]  # (T, N, 3) in nm
        xyz_path_A = xyz_path_nm * NM_TO_ANG  # Convert to Angstroms
        
        for t in range(T):
            xyz_pred_A = xyz_path_A[t].astype(np.float64)
            
            # Kabsch align to reference structure
            try:
                xyz_pred_aligned_A, _ = kabsch_align_np(xyz_pred_A, ref_xyz_A)
            except Exception as e:
                print(f"[warn] Kabsch alignment failed for frame {t}: {e}")
                xyz_pred_aligned_A = xyz_pred_A
            
            # Write PDB with aligned coordinates
            ag.positions = xyz_pred_aligned_A.astype(np.float32)
            ag.write(str(pdb_dir_b / f"pred_frame_{t:05d}.pdb"))
        
        print(f"[saved] Path {b} PDBs: {pdb_dir_b}")
        
        # ====================================================================
        # Compute MolProbity Metrics (following ensemble_gen.py)
        # ====================================================================
        print(f"[molprobity] Computing structure quality metrics for path {b}...")
        metrics = compute_structure_metrics_all_frames(str(pdb_dir_b))
        
        if metrics:
            # Add path identifier
            for m in metrics:
                m['path'] = b
            
            # Compute inter-frame RMSD in Angstroms
            interframe_rmsds = compute_interframe_rmsd(xyz_path_A)
            
            # Add RMSD statistics to first metric entry
            if len(metrics) > 0:
                metrics[0]['interframe_rmsd_mean'] = float(np.mean(interframe_rmsds)) if len(interframe_rmsds) > 0 else np.nan
                metrics[0]['interframe_rmsd_std'] = float(np.std(interframe_rmsds)) if len(interframe_rmsds) > 0 else np.nan
                metrics[0]['interframe_rmsd_max'] = float(np.max(interframe_rmsds)) if len(interframe_rmsds) > 0 else np.nan
            
            all_metrics.extend(metrics)
        else:
            print(f"[warn] No metrics computed for path {b}")
        
        # ====================================================================
        # Visualize Latent Space with Path
        # ====================================================================
        viz_path = out_dir / f"{prefix}_neb_path_{b:02d}_latent_viz.png"
        path_latents_np = traj[:, b, :].detach().cpu().numpy()  # (T, D)
        
        visualize_latent_space_with_path(
            latents_all=latents_all,
            path_latents=path_latents_np,
            full_dataset=full_dataset,
            start_idx=start_idx,
            end_idx=end_idx,
            out_path=str(viz_path),
            protein_name=args.protein,
            path_idx=b
        )
    
    # ========================================================================
    # Save Metrics to CSV and Print Summary (following ensemble_gen.py)
    # ========================================================================
    if all_metrics:
        metrics_csv = out_dir / f"{prefix}_neb_molprobity_scores.csv"
        df = pd.DataFrame(all_metrics)
        df.to_csv(metrics_csv, index=False)
        print(f"\n[saved] MolProbity scores saved to: {metrics_csv}")
        
        # Print summary statistics PER PATH (following ensemble_gen.py format)
        print("\n" + "="*80)
        print("MolProbity Score Summary (per path)")
        print("="*80)
        
        for b in range(B):
            path_df = df[df['path'] == b]
            if len(path_df) == 0:
                continue
            
            print(f"\nPath {b}:")
            print(f"{'Metric':<40} {'Mean ± Std Dev':<20}")
            print("-"*80)
            
            if 'rama_outliers_pct' in path_df.columns:
                mean = path_df['rama_outliers_pct'].mean()
                std = path_df['rama_outliers_pct'].std()
                print(f"{'Ramachandran Outliers (%)':<40} {mean:>6.2f} ± {std:<6.2f}")
            
            if 'clashscore' in path_df.columns:
                mean = path_df['clashscore'].mean()
                std = path_df['clashscore'].std()
                print(f"{'Clashscore':<40} {mean:>6.2f} ± {std:<6.2f}")
            
            if 'rotamer_outliers_pct' in path_df.columns:
                mean = path_df['rotamer_outliers_pct'].mean()
                std = path_df['rotamer_outliers_pct'].std()
                print(f"{'Rotamer Outliers (%)':<40} {mean:>6.2f} ± {std:<6.2f}")
            
            if 'molprobity_score' in path_df.columns:
                mean = path_df['molprobity_score'].mean()
                std = path_df['molprobity_score'].std()
                print(f"{'MolProbity Score':<40} {mean:>6.2f} ± {std:<6.2f}")
            
            # Add inter-frame RMSD if available
            if 'interframe_rmsd_mean' in path_df.columns:
                rmsd_mean = path_df['interframe_rmsd_mean'].iloc[0]
                rmsd_std = path_df['interframe_rmsd_std'].iloc[0]
                rmsd_max = path_df['interframe_rmsd_max'].iloc[0]
                print(f"{'Inter-frame RMSD (Å) - Mean':<40} {rmsd_mean:>6.3f}")
                print(f"{'Inter-frame RMSD (Å) - Std Dev':<40} {rmsd_std:>6.3f}")
                print(f"{'Inter-frame RMSD (Å) - Max':<40} {rmsd_max:>6.3f}")
        
        print("="*80)
    else:
        print("[warn] No metrics computed")
    
    # ========================================================================
    # Save Payload
    # ========================================================================
    payload: Dict[str, Any] = {
        "traj": traj.detach().cpu().numpy(),
        "protein": args.protein,
        "pdb_path": pdb_path,
        "latent_path": args.latent_path,
        "model_path": args.model_path,
        "method": "NEB",
        "n_pivots": getattr(args, 'n_pivots', 20),
        "neb_steps": getattr(args, 'neb_steps', 50),
        "neb_lr": getattr(args, 'neb_lr', 0.05),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "n_paths": n_paths,
        "energy_mu": float(energy_mu),
        "energy_sd": float(energy_sd),
        "metrics": all_metrics,
        "pdb_dir": str(base_pdb_dir),
        "metrics_csv": str(metrics_csv) if all_metrics else None,
    }

    with open(args.out, "wb") as outfh:
        pickle.dump(payload, outfh)
    print(f"\n[saved] Payload pkl: {os.path.abspath(args.out)}")
    
    print("\n" + "="*80)
    print("NEB Path Generation Complete!")
    print(f"Results saved to: {out_dir}")
    print("="*80)
    print("\nPyMOL tip:")
    print(f"  cd {base_pdb_dir / 'path_00'}")
    print("  load pred_frame_00000.pdb")
    print("  load pred_frame_00001.pdb")
    print("="*80)


if __name__ == "__main__":
    main()

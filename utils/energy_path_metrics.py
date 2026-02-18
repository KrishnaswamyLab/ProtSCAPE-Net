"""
Utilities for computing energy metrics along conformational paths from PDB files.
"""

import os
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
import MDAnalysis as mda


def compute_energies_from_pdbs(
    pdb_paths: List[str],
    setup_simulation_fn: Callable,
    reference_pdb: str,
    sel_atom_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute OpenMM potential energies for a list of PDB files.
    
    Args:
        pdb_paths: List of paths to PDB files (in sequential order)
        setup_simulation_fn: Function that creates OpenMM simulation from a PDB path
        reference_pdb: Path to reference PDB file for OpenMM system setup
        sel_atom_indices: Optional atom indices into the full reference system used
            when path PDBs contain only a selected subset of atoms
        
    Returns:
        Array of energies in kJ/mol, shape (n_frames,)
    """
    from openmm.unit import nanometer, kilojoule_per_mole
    
    # Setup OpenMM simulation once using reference PDB
    sim_out = setup_simulation_fn(reference_pdb)
    simulation = sim_out[0] if isinstance(sim_out, tuple) else sim_out

    # Reference full coordinates (used to reconstruct full system coordinates
    # if path PDBs contain only selected atoms)
    u_ref = mda.Universe(reference_pdb)
    full_ref_nm = u_ref.atoms.positions.astype(np.float64) / 10.0

    expected_n_atoms = None
    try:
        expected_n_atoms = simulation.topology.getNumAtoms()
    except Exception:
        try:
            expected_n_atoms = simulation.context.getSystem().getNumParticles()
        except Exception:
            expected_n_atoms = None

    if expected_n_atoms is not None and expected_n_atoms != full_ref_nm.shape[0]:
        raise RuntimeError(
            f"Reference PDB has {full_ref_nm.shape[0]} atoms, but OpenMM expects {expected_n_atoms}."
        )

    if sel_atom_indices is not None:
        sel_atom_indices = np.asarray(sel_atom_indices, dtype=int).reshape(-1)
        if np.any(sel_atom_indices < 0) or np.any(sel_atom_indices >= full_ref_nm.shape[0]):
            raise ValueError("sel_atom_indices contains out-of-range indices for reference_pdb atoms")
    
    energies = []
    
    for pdb_path in pdb_paths:
        try:
            # Load coordinates from PDB
            u = mda.Universe(pdb_path)
            positions_A = u.atoms.positions.astype(np.float64)
            
            # Convert Angstroms to nanometers
            positions_nm = positions_A / 10.0

            n_frame_atoms = positions_nm.shape[0]
            n_expected = expected_n_atoms if expected_n_atoms is not None else full_ref_nm.shape[0]

            if n_frame_atoms == n_expected:
                full_positions_nm = positions_nm
            elif sel_atom_indices is not None and n_frame_atoms == sel_atom_indices.shape[0]:
                full_positions_nm = full_ref_nm.copy()
                full_positions_nm[sel_atom_indices] = positions_nm
            else:
                raise RuntimeError(
                    f"Atom-count mismatch for {os.path.basename(pdb_path)}: frame has {n_frame_atoms} atoms, "
                    f"OpenMM expects {n_expected}, selection has "
                    f"{None if sel_atom_indices is None else sel_atom_indices.shape[0]}."
                )
            
            # Set positions in OpenMM context
            from openmm import unit
            qpos = unit.Quantity(full_positions_nm, unit=nanometer)
            simulation.context.setPositions(qpos)
            
            # Get potential energy
            state = simulation.context.getState(getEnergy=True)
            pe = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
            
            energies.append(float(pe))
            
        except Exception as e:
            print(f"[warn] Failed to compute energy for {pdb_path}: {str(e)[:100]}")
            energies.append(np.nan)
    
    return np.array(energies)


def compute_energy_integral(energies: np.ndarray, method: str = 'trapezoid') -> float:
    """
    Compute the integral/summation of energy along a path.
    
    Args:
        energies: Energy values along path, shape (n_frames,)
        method: Integration method - 'trapezoid' (default) or 'sum'
        
    Returns:
        Integrated energy value
    """
    # Remove NaN values for computation
    valid_energies = energies[~np.isnan(energies)]
    
    if len(valid_energies) == 0:
        return np.nan
    
    if method == 'trapezoid':
        # Trapezoidal rule for numerical integration
        # Assumes uniform spacing between frames
        integral = np.trapz(valid_energies)
    elif method == 'sum':
        # Simple summation
        integral = np.sum(valid_energies)
    else:
        raise ValueError(f"Unknown integration method: {method}. Use 'trapezoid' or 'sum'")
    
    return float(integral)


def compute_energy_metrics_from_pdb_dir(
    pdb_dir: str,
    setup_simulation_fn: Callable,
    reference_pdb: str,
    sel_atom_indices: Optional[np.ndarray] = None,
    normalize_by_start: bool = True,
) -> Dict[str, float]:
    """
    Compute comprehensive energy metrics for all PDB files in a directory.
    
    Args:
        pdb_dir: Directory containing PDB files
        setup_simulation_fn: Function that creates OpenMM simulation
        reference_pdb: Path to reference PDB file
        sel_atom_indices: Optional atom indices into the full reference system used
            when path PDBs contain only selected atoms
        normalize_by_start: If True, normalize energies as ΔE relative to the
            first valid frame so metrics are comparable across paths
        
    Returns:
        Dictionary with energy metrics including integral
    """
    # Get all PDB files in directory, sorted
    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])
    
    if len(pdb_files) == 0:
        print("[warn] No PDB files found in directory")
        return {
            'energy_mean': np.nan,
            'energy_std': np.nan,
            'energy_min': np.nan,
            'energy_max': np.nan,
            'energy_integral_trapz': np.nan,
            'energy_integral_sum': np.nan,
            'energy_start': np.nan,
            'energy_end': np.nan,
            'n_frames': 0,
        }
    
    # Get full paths
    pdb_paths = [os.path.join(pdb_dir, f) for f in pdb_files]
    
    print(f"[energy] Computing energies for {len(pdb_paths)} PDB files...")
    
    # Compute energies
    energies = compute_energies_from_pdbs(
        pdb_paths,
        setup_simulation_fn,
        reference_pdb,
        sel_atom_indices=sel_atom_indices,
    )

    energies_used = energies.copy()
    if normalize_by_start:
        valid_idx = np.where(~np.isnan(energies_used))[0]
        if len(valid_idx) > 0:
            baseline = energies_used[valid_idx[0]]
            energies_used = energies_used - baseline
        else:
            baseline = np.nan
    else:
        baseline = np.nan
    
    # Compute integral using both methods
    integral_trapz = compute_energy_integral(energies_used, method='trapezoid')
    integral_sum = compute_energy_integral(energies_used, method='sum')
    
    # Remove NaN for statistics
    valid_energies = energies_used[~np.isnan(energies_used)]
    
    # Compute metrics
    metrics = {
        'energy_mean': float(np.mean(valid_energies)) if len(valid_energies) > 0 else np.nan,
        'energy_std': float(np.std(valid_energies)) if len(valid_energies) > 0 else np.nan,
        'energy_min': float(np.min(valid_energies)) if len(valid_energies) > 0 else np.nan,
        'energy_max': float(np.max(valid_energies)) if len(valid_energies) > 0 else np.nan,
        'energy_integral_trapz': integral_trapz,
        'energy_integral_sum': integral_sum,
        'energy_start': float(energies_used[0]) if len(energies_used) > 0 else np.nan,
        'energy_end': float(energies_used[-1]) if len(energies_used) > 0 else np.nan,
        'n_frames': len(pdb_files),
        'n_valid_frames': len(valid_energies),
    }

    if normalize_by_start:
        metrics['energy_baseline_abs_kjmol'] = float(baseline)
        metrics['energy_normalization'] = 'delta_from_first_valid'
    
    print(f"[energy] Computed energy metrics for {metrics['n_valid_frames']}/{metrics['n_frames']} frames")
    if normalize_by_start:
        print("[energy] Reporting normalized energies as ΔE relative to first valid frame")
    print(f"[energy] Mean energy: {metrics['energy_mean']:.2f} kJ/mol")
    print(f"[energy] Energy integral (trapezoid): {metrics['energy_integral_trapz']:.2f}")
    print(f"[energy] Energy integral (sum): {metrics['energy_integral_sum']:.2f}")
    
    return metrics

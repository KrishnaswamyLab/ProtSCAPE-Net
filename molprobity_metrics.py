
import subprocess
import re
import os
import shutil
import argparse
import numpy as np
try:
    from Bio.PDB import PDBParser
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

try:
    from utils.energy_path_metrics import compute_energy_metrics_from_pdb_dir
    ENERGY_METRICS_AVAILABLE = True
except ImportError:
    ENERGY_METRICS_AVAILABLE = False
    print("[warn] Energy path metrics not available. Install utils.energy_path_metrics to compute energy.")

try:
    from utils.generation_helpers import load_module_from_path
    MODULE_LOADER_AVAILABLE = True
except ImportError:
    MODULE_LOADER_AVAILABLE = False
    print("[warn] Module loader not available. Install utils.generation_helpers to load setup_simulation.")

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


def extract_ca_coords_from_pdb(pdb_path: str) -> np.ndarray:
    """
    Extract C-alpha coordinates from a PDB file.
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        Coordinates array of shape (N, 3) for N C-alpha atoms
    """
    if not BIOPYTHON_AVAILABLE:
        return None
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
            break  # Only use first model
        
        return np.array(coords) if coords else None
    except Exception:
        return None


def compute_interframe_rmsd_from_pdbs(pdb_paths: list) -> np.ndarray:
    """
    Compute RMSD in Angstroms between consecutive PDB frames.
    
    Args:
        pdb_paths: List of paths to PDB files in order
        
    Returns:
        Array of RMSDs between consecutive frames, shape (T-1,)
    """
    if len(pdb_paths) < 2:
        return np.array([])
    
    # Extract coordinates from all frames
    coords_list = []
    for pdb_path in pdb_paths:
        coords = extract_ca_coords_from_pdb(pdb_path)
        if coords is not None:
            coords_list.append(coords)
        else:
            return np.array([])  # Return empty if any frame fails
    
    # Compute RMSD between consecutive frames
    rmsds = []
    for i in range(len(coords_list) - 1):
        xyz1 = coords_list[i]
        xyz2 = coords_list[i + 1]
        
        # Check if same number of atoms
        if xyz1.shape[0] != xyz2.shape[0]:
            rmsds.append(np.nan)
            continue
        
        diff = xyz2 - xyz1
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))
        rmsds.append(rmsd)
    
    return np.array(rmsds)


def compute_structure_metrics_all_frames(pdb_dir: str, 
                                         setup_simulation_fn=None,
                                         reference_pdb: str = None):
    """
    Compute comprehensive MolProbity scores for all PDB frames.
    
    Args:
        pdb_dir: Directory containing PDB files
        setup_simulation_fn: Optional function to create OpenMM simulation for energy computation
        reference_pdb: Optional path to reference PDB for OpenMM energy computation
        
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
        # return None
    
    print("[molprobity] Using phenix.molprobity for scoring")
    
    # Compute interframe RMSD if BioPython is available
    interframe_rmsds = None
    if BIOPYTHON_AVAILABLE:
        print("[rmsd] Computing interframe RMSD...")
        pdb_paths = [os.path.join(pdb_dir, f) for f in pdbs]
        interframe_rmsds = compute_interframe_rmsd_from_pdbs(pdb_paths)
        if len(interframe_rmsds) > 0:
            print(f"[rmsd] Computed {len(interframe_rmsds)} interframe RMSDs")
    
    # Compute energy metrics if OpenMM setup is provided
    energy_metrics = None
    import pdb; pdb.set_trace()  # Debug: Check if setup_simulation_fn and reference_pdb are provided
    if setup_simulation_fn is not None and reference_pdb is not None and ENERGY_METRICS_AVAILABLE:
        print("[energy] Computing energy metrics from PDB files...")
        try:
            energy_metrics = compute_energy_metrics_from_pdb_dir(
                pdb_dir=pdb_dir,
                setup_simulation_fn=setup_simulation_fn,
                reference_pdb=reference_pdb,
            )
        except Exception as e:
            print(f"[warn] Failed to compute energy metrics: {str(e)[:100]}")
            energy_metrics = None
    
    all_metrics = []
    
    for i, pdb_file in enumerate(pdbs):
        pdb_path = os.path.join(pdb_dir, pdb_file)
        
        try:
            # Use phenix.molprobity
            result = run_phenix_molprobity(pdb_path)
            
            # Get interframe RMSD for this frame (if available)
            rmsd_to_next = np.nan
            if interframe_rmsds is not None and i < len(interframe_rmsds):
                rmsd_to_next = interframe_rmsds[i]
            
            if result:
                metrics = {
                    'frame': i,
                    'pdb_file': pdb_file,
                    'rama_outliers_pct': result.get('ramachandran_outliers', np.nan),
                    'clashscore': result.get('clashscore', np.nan),
                    'rotamer_outliers_pct': result.get('rotamer_outliers', np.nan),
                    'molprobity_score': result.get('molprobity_score', np.nan),
                    'rmsd_to_next_frame': rmsd_to_next,
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
                    'rmsd_to_next_frame': rmsd_to_next,
                }
            
            all_metrics.append(metrics)
            
            if (i + 1) % 10 == 0:
                print(f"[progress] Scored {i + 1}/{len(pdbs)} frames")
                
        except Exception as e:
            print(f"[warn] Failed to score {pdb_file}: {str(e)[:100]}")
            rmsd_to_next = np.nan
            if interframe_rmsds is not None and i < len(interframe_rmsds):
                rmsd_to_next = interframe_rmsds[i]
            all_metrics.append({
                'frame': i,
                'pdb_file': pdb_file,
                'rama_outliers_pct': np.nan,
                'clashscore': np.nan,
                'rotamer_outliers_pct': np.nan,
                'molprobity_score': np.nan,
                'rmsd_to_next_frame': rmsd_to_next,
            })
    
    # Add energy metrics to the first frame's entry if available
    if energy_metrics and len(all_metrics) > 0:
        all_metrics[0].update({
            'energy_mean': energy_metrics.get('energy_mean', np.nan),
            'energy_std': energy_metrics.get('energy_std', np.nan),
            'energy_min': energy_metrics.get('energy_min', np.nan),
            'energy_max': energy_metrics.get('energy_max', np.nan),
            'energy_integral_trapz': energy_metrics.get('energy_integral_trapz', np.nan),
            'energy_integral_sum': energy_metrics.get('energy_integral_sum', np.nan),
            'energy_start': energy_metrics.get('energy_start', np.nan),
            'energy_end': energy_metrics.get('energy_end', np.nan),
        })
    
    return all_metrics


def compute_average_metrics(all_metrics: list):
    """
    Compute average metrics across all frames.
    
    Args:
        all_metrics: List of dictionaries with metrics for each frame
        
    Returns:
        Dictionary with average values for each metric
    """
    if not all_metrics:
        return None
    
    # Extract metric values
    rama_outliers = [m['rama_outliers_pct'] for m in all_metrics if not np.isnan(m['rama_outliers_pct'])]
    clashscores = [m['clashscore'] for m in all_metrics if not np.isnan(m['clashscore'])]
    rotamer_outliers = [m['rotamer_outliers_pct'] for m in all_metrics if not np.isnan(m['rotamer_outliers_pct'])]
    molprobity_scores = [m['molprobity_score'] for m in all_metrics if not np.isnan(m['molprobity_score'])]
    interframe_rmsds = [m['rmsd_to_next_frame'] for m in all_metrics if not np.isnan(m['rmsd_to_next_frame'])]
    
    avg_metrics = {
        'avg_rama_outliers_pct': np.mean(rama_outliers) if rama_outliers else np.nan,
        'avg_clashscore': np.mean(clashscores) if clashscores else np.nan,
        'avg_rotamer_outliers_pct': np.mean(rotamer_outliers) if rotamer_outliers else np.nan,
        'avg_molprobity_score': np.mean(molprobity_scores) if molprobity_scores else np.nan,
        'avg_interframe_rmsd': np.mean(interframe_rmsds) if interframe_rmsds else np.nan,
        'std_rama_outliers_pct': np.std(rama_outliers) if rama_outliers else np.nan,
        'std_clashscore': np.std(clashscores) if clashscores else np.nan,
        'std_rotamer_outliers_pct': np.std(rotamer_outliers) if rotamer_outliers else np.nan,
        'std_molprobity_score': np.std(molprobity_scores) if molprobity_scores else np.nan,
        'std_interframe_rmsd': np.std(interframe_rmsds) if interframe_rmsds else np.nan,
        'n_frames': len(all_metrics),
        'n_valid_frames': len([m for m in all_metrics if not np.isnan(m['molprobity_score'])])
    }
    
    # Add energy metrics if available in the first frame
    if len(all_metrics) > 0 and 'energy_mean' in all_metrics[0]:
        avg_metrics.update({
            'energy_mean': all_metrics[0].get('energy_mean', np.nan),
            'energy_std': all_metrics[0].get('energy_std', np.nan),
            'energy_min': all_metrics[0].get('energy_min', np.nan),
            'energy_max': all_metrics[0].get('energy_max', np.nan),
            'energy_integral_trapz': all_metrics[0].get('energy_integral_trapz', np.nan),
            'energy_integral_sum': all_metrics[0].get('energy_integral_sum', np.nan),
            'energy_start': all_metrics[0].get('energy_start', np.nan),
            'energy_end': all_metrics[0].get('energy_end', np.nan),
        })
    
    return avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MolProbity + energy metrics for baseline/generated PDB frames")
    parser.add_argument("--pdb_dir", type=str, default="baselines/7lp1", help="Directory containing frame PDB files")
    parser.add_argument("--atlas_rt_path", type=str, default="data/prepare_atlas.py", help="Path to atlas runtime module defining setup_simulation")
    parser.add_argument("--reference_pdb", type=str, default="data/datasets/7lp1_A_protein/7lp1_A.pdb", help="Reference full-system PDB for OpenMM energy")
    args = parser.parse_args()

    setup_simulation_fn = None
    if ENERGY_METRICS_AVAILABLE:
        if not MODULE_LOADER_AVAILABLE:
            print("[warn] Skipping energy metrics: module loader unavailable")
        elif not os.path.exists(args.atlas_rt_path):
            print(f"[warn] Skipping energy metrics: atlas_rt_path not found: {args.atlas_rt_path}")
        elif not os.path.exists(args.reference_pdb):
            print(f"[warn] Skipping energy metrics: reference_pdb not found: {args.reference_pdb}")
        else:
            try:
                atlas_rt = load_module_from_path(args.atlas_rt_path, module_name="atlas_new")
                if hasattr(atlas_rt, "setup_simulation"):
                    setup_simulation_fn = getattr(atlas_rt, "setup_simulation")
                    print(f"[energy] Loaded setup_simulation from {args.atlas_rt_path}")
                    print(f"[energy] Using reference_pdb: {args.reference_pdb}")
                else:
                    print(f"[warn] Skipping energy metrics: {args.atlas_rt_path} does not define setup_simulation")
            except Exception as e:
                print(f"[warn] Skipping energy metrics: failed to load atlas module ({str(e)[:120]})")

    all_metrics = compute_structure_metrics_all_frames(
        args.pdb_dir,
        setup_simulation_fn=setup_simulation_fn,
        reference_pdb=args.reference_pdb if setup_simulation_fn is not None else None,
    )
    
    if all_metrics:
        print("\n" + "="*60)
        print("AVERAGE METRICS ACROSS ALL FRAMES")
        print("="*60)
        avg_metrics = compute_average_metrics(all_metrics)
        for key, value in avg_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        print("="*60)
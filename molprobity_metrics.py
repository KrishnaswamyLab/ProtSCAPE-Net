
import subprocess
import re
import os
import shutil
import numpy as np
try:
    from Bio.PDB import PDBParser
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

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
    
    # Compute interframe RMSD if BioPython is available
    interframe_rmsds = None
    if BIOPYTHON_AVAILABLE:
        print("[rmsd] Computing interframe RMSD...")
        pdb_paths = [os.path.join(pdb_dir, f) for f in pdbs]
        interframe_rmsds = compute_interframe_rmsd_from_pdbs(pdb_paths)
        if len(interframe_rmsds) > 0:
            print(f"[rmsd] Computed {len(interframe_rmsds)} interframe RMSDs")
    
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
    
    return avg_metrics


if __name__ == "__main__":
    all_metrics = compute_structure_metrics_all_frames("baselines/6h86")
    
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
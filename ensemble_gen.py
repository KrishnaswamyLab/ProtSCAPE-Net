"""
Main Pipeline Script.
Runs the complete Latent Diffusion Pipeline:
1. Autoencoder Training (Compression)
2. DDPM Training (Generation)
3. Sample Generation & Evaluation
4. Decode to coordinates and export PDBs

Usage:
  python ensemble_gen.py --config configs/config_ensemble.yaml
  python ensemble_gen.py --config configs/config_ensemble.yaml --skip_ae --skip_ddpm
"""
import argparse
import sys
import os
import pickle
import glob
from pathlib import Path
import numpy as np
import torch

# Add current directory to path just in case
sys.path.append(str(Path(__file__).parent))

from protscape.config import Config
from utils.config import load_config
from protscape.train_ae import train_ae
from protscape.train_ddpm import train_ddpm
from protscape.generate import generate
from protscape.protscape import ProtSCAPE
from utils.generation_helpers import pick_default_pdb
from utils.geometry import kabsch_align_np
from utils.generation_viz import compute_structure_metrics
import MDAnalysis as mda
import pandas as pd

NM_TO_ANG = 10.0


def compute_structure_metrics_all_frames(pdb_dir: str):
    """
    Compute comprehensive MolProbity scores for all PDB frames using phenix.molprobity.
    
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


def check_phenix_available() -> bool:
    """
    Check if phenix.molprobity is available.
    
    Returns:
        True if phenix.molprobity is available
    """
    try:
        import subprocess
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
        import subprocess
        import re
        
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


def decode_and_export_pdbs(config):
    """
    Decode generated latent samples to coordinates and export as PDB files.
    
    Args:
        config: Config object with paths and settings
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    
    # Load generated samples
    generated_samples_path = Path(config.output_dir) / "generated_samples.npy"
    if not generated_samples_path.exists():
        raise FileNotFoundError(f"Generated samples not found at {generated_samples_path}")
    
    generated_latents = np.load(generated_samples_path).astype(np.float32)
    print(f"[loaded] Generated latents shape: {generated_latents.shape}")
    
    # Limit to n_pdb_samples
    n_samples = min(config.n_pdb_samples, generated_latents.shape[0])
    generated_latents = generated_latents[:n_samples]
    print(f"[subset] Using {n_samples} samples for PDB export")
    
    # Load dataset to get model parameters
    protein_id = config.protein.lower()
    dataset_path = config.dataset_path
    
    # Try pattern: {protein_id}_C_graphs.pkl
    pattern = f"{protein_id}_C_graphs.pkl"
    matches = glob.glob(os.path.join(dataset_path, pattern))
    
    if matches:
        dataset_path = matches[0]
        print(f"[dataset] auto-detected: {dataset_path}")
    else:
        raise FileNotFoundError(
            f"No dataset found matching pattern '{pattern}'. "
            f"Expected format: {{pdbid}}_C_graphs.pkl"
        )
    
    with open(dataset_path, "rb") as f:
        full_dataset = pickle.load(f)
    
    if len(full_dataset) == 0:
        raise RuntimeError("Dataset is empty.")
    
    # Setup model parameters from dataset
    class ModelArgs:
        pass
    
    model_args = ModelArgs()
    model_args.num_nodes = full_dataset[0].x.shape[0]
    model_args.node_feat_dim = full_dataset[0].x.shape[1]
    model_args.input_dim = 3
    model_args.prot_graph_size = model_args.num_nodes
    model_args.latent_dim = generated_latents.shape[1]
    
    Z_max = int(max(g.x[:, 0].max().item() for g in full_dataset))
    res_max = int(max(g.x[:, 1].max().item() for g in full_dataset))
    aa_max = int(max(g.x[:, 2].max().item() for g in full_dataset))
    
    model_args.num_Z = Z_max + 1
    model_args.num_residues = res_max + 1
    model_args.num_aa = max(aa_max + 1, 21)
    
    # Add all required ProtSCAPE hyperparameters
    model_args.hidden_dim = 256
    model_args.embedding_dim = 128
    model_args.lr = 1e-3
    model_args.alpha = 50.0
    model_args.beta_loss = 0.5
    model_args.coord_weight = 50.0
    model_args.probs = 0.2
    model_args.nhead = 1
    model_args.layers = 3
    model_args.task = "reg"
    model_args.num_mp_layers = 2
    model_args.mp_hidden = 256

    # Add ablation configuration for feature extractors and node/edge features
    if hasattr(config, 'feature_extractor'):
        model_args.feature_extractor = config.feature_extractor
    else:
        model_args.feature_extractor = "scattering"  # default
    
    if hasattr(config, 'gcn_num_layers'):
        model_args.gcn_num_layers = config.gcn_num_layers
    else:
        model_args.gcn_num_layers = 4
    
    if hasattr(config, 'gcn_hidden_channels'):
        model_args.gcn_hidden_channels = config.gcn_hidden_channels
    else:
        model_args.gcn_hidden_channels = None
    
    if hasattr(config, 'ablate_node_features'):
        model_args.ablate_node_features = config.ablate_node_features
    if hasattr(config, 'ablate_edge_features'):
        model_args.ablate_edge_features = config.ablate_edge_features
    
    print(f"[model] num_nodes={model_args.num_nodes}, latent_dim={model_args.latent_dim}")
    
    # Load ProtSCAPE model
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model not found at {config.model_path}")
    
    model = ProtSCAPE(model_args).to(device).eval()
    weights = torch.load(config.model_path, map_location=device)
    model.load_state_dict(weights)
    print(f"[model] Loaded ProtSCAPE from {config.model_path}")
    
    # Decode latents to xyz coordinates (in nm)
    print("[decode] Decoding latents to xyz coordinates...")
    latents_tensor = torch.from_numpy(generated_latents).to(device)
    
    xyz_nm_list = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, latents_tensor.shape[0], batch_size):
            batch = latents_tensor[i:i+batch_size]
            xyz_batch = model.reconstruct_xyz(batch)
            xyz_np = xyz_batch.detach().cpu().numpy()
            
            # Ensure shape is (batch, N, 3)
            if xyz_np.ndim == 3 and xyz_np.shape[-1] == 3:
                xyz_nm_list.append(xyz_np)
            else:
                xyz_reshaped = xyz_np.reshape(xyz_np.shape[0], -1, 3)
                xyz_nm_list.append(xyz_reshaped)
    
    xyz_nm = np.concatenate(xyz_nm_list, axis=0)  # (n_samples, N, 3)
    print(f"[decode] Decoded xyz shape: {xyz_nm.shape} (in nm)")
    
    # Load normalization parameters if needed
    xyz_mu = xyz_sd = None
    if config.normalize_xyz:
        if config.xyz_mu_path and config.xyz_sd_path:
            xyz_mu = np.load(config.xyz_mu_path)
            xyz_sd = np.load(config.xyz_sd_path)
            print(f"[decode] Loaded normalization params from {config.xyz_mu_path} and {config.xyz_sd_path}")
            # Denormalize: xyz = xyz * sd + mu (nm)
            xyz_nm = xyz_nm * xyz_sd + xyz_mu
            print("[decode] Applied denormalization: xyz = xyz*sd + mu")
        else:
            print("[warn] normalize_xyz=True but no xyz_mu_path/xyz_sd_path provided, skipping denormalization")
    
    # Get PDB path
    pdb_path = config.pdb or pick_default_pdb(config.protein)
    if pdb_path is None or not os.path.exists(pdb_path):
        raise FileNotFoundError("No PDB found. Provide --pdb or place a PDB in a known analysis folder.")
    print(f"[pdb] Using reference PDB: {pdb_path}")
    
    # Get selection indices if available
    sel_idx = None
    if hasattr(full_dataset[0], "sel_atom_indices"):
        sel_idx = full_dataset[0].sel_atom_indices.detach().cpu().numpy().astype(int)
        print(f"[decode] Using selection indices: {len(sel_idx)} atoms")
    
    # Setup MDAnalysis for PDB export (following inference.py approach)
    u = mda.Universe(pdb_path)
    
    if sel_idx is None:
        raise RuntimeError(
            "Dataset graphs do not have sel_atom_indices. "
            "Please store sel_atom_indices during preprocessing so export ordering matches."
        )
    
    ag = u.atoms[sel_idx]
    print(f"[info] Using stored sel_atom_indices for export: {len(ag)} atoms")
    
    if len(ag) != model_args.num_nodes:
        raise ValueError(
            f"MDAnalysis AtomGroup has {len(ag)} atoms but graphs have {model_args.num_nodes} nodes. "
            "Counts must match for PDB export."
        )
    
    # Get reference structure in Å for alignment
    ref_xyz_A = ag.positions.astype(np.float64)
    
    # Export ALL PDBs (not just limited samples)
    pdb_output_dir = Path(config.output_dir) / "generated_pdbs"
    os.makedirs(pdb_output_dir, exist_ok=True)
    print(f"[export] Exporting ALL {xyz_nm.shape[0]} PDB files to {pdb_output_dir}")
    
    # Export all frames
    for i in range(xyz_nm.shape[0]):
        # Convert nm to Å
        xyz_pred_A = (xyz_nm[i] * NM_TO_ANG).astype(np.float64)
        
        # Kabsch align to reference structure
        try:
            xyz_pred_aligned_A, _ = kabsch_align_np(xyz_pred_A, ref_xyz_A)
        except Exception as e:
            print(f"[warn] Kabsch alignment failed for sample {i}: {e}")
            xyz_pred_aligned_A = xyz_pred_A
        
        # Write PDB with aligned coordinates
        ag.positions = xyz_pred_aligned_A.astype(np.float32)
        ag.write(str(pdb_output_dir / f"pred_frame_{i:05d}.pdb"))
    
    print(f"[saved] PDB files exported to: {pdb_output_dir}")
    
    # Compute MolProbity scores for all frames
    print(f"\n[molprobity] Computing structure quality metrics for all {xyz_nm.shape[0]} frames...")
    metrics = compute_structure_metrics_all_frames(str(pdb_output_dir))
    
    if metrics:
        # Save metrics to CSV
        metrics_csv = Path(config.output_dir) / "molprobity_scores.csv"
        df = pd.DataFrame(metrics)
        df.to_csv(metrics_csv, index=False)
        print(f"[saved] MolProbity scores saved to: {metrics_csv}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("MolProbity Score Summary (all frames)")
        print("="*80)
        print(f"{'Metric':<40} {'Mean ± Std Dev':<20}")
        print("-"*80)
        
        if 'rama_outliers_pct' in df.columns:
            mean = df['rama_outliers_pct'].mean()
            std = df['rama_outliers_pct'].std()
            print(f"{'Ramachandran Outliers (%)':<40} {mean:>6.2f} ± {std:<6.2f}")
        
        if 'clashscore' in df.columns:
            mean = df['clashscore'].mean()
            std = df['clashscore'].std()
            print(f"{'Clashscore':<40} {mean:>6.2f} ± {std:<6.2f}")
        
        if 'rotamer_outliers_pct' in df.columns:
            mean = df['rotamer_outliers_pct'].mean()
            std = df['rotamer_outliers_pct'].std()
            print(f"{'Rotamer Outliers (%)':<40} {mean:>6.2f} ± {std:<6.2f}")
        
        if 'molprobity_score' in df.columns:
            mean = df['molprobity_score'].mean()
            std = df['molprobity_score'].std()
            print(f"{'MolProbity Score':<40} {mean:>6.2f} ± {std:<6.2f}")
        
        print("="*80)
    else:
        print("[warn] No metrics computed")
    
    print(f"\n[done] Decoding and PDB export completed!")
    print("\nPyMOL tip:")
    print(f"  cd {pdb_output_dir}")
    print("  load pred_frame_00000.pdb")
    print("  load pred_frame_00001.pdb")


def main():
    parser = argparse.ArgumentParser(
        description="Run Hierarchical Latent Diffusion Pipeline for Conformational Ensemble Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Edit configs/config_ensemble.yaml, then run:
  python ensemble_gen.py --config configs/config_ensemble.yaml
  
  # Skip training stages (reuse existing checkpoints):
  python ensemble_gen.py --config configs/config_ensemble.yaml --skip_ae --skip_ddpm
  
  # Override specific parameters:
  python ensemble_gen.py --config configs/config_ensemble.yaml --n_pdb_samples 200 --epochs 500

See configs/ENSEMBLE_CONFIG_GUIDE.md for detailed usage instructions.
        """
    )
    
    # Config file argument
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file (configs/config_ensemble.yaml)")
    
    # Training control
    parser.add_argument("--skip_ae", action="store_true", help="Skip Autoencoder training")
    parser.add_argument("--skip_ddpm", action="store_true", help="Skip DDPM training")
    
    # Optional overrides for key parameters
    parser.add_argument("--protein", type=str, default=None, help="Override protein ID")
    parser.add_argument("--n_pdb_samples", type=int, default=None, help="Override number of PDB samples to export")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config_dict = load_config(args.config)
    
    # Initialize Config from dictionary
    config = Config(**config_dict)
    
    # Apply command-line overrides
    if args.protein is not None:
        config.protein = args.protein
    if args.n_pdb_samples is not None:
        config.n_pdb_samples = args.n_pdb_samples
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.epochs is not None:
        config.epochs = args.epochs
    
    print("="*80)
    print("ProtSCAPE Ensemble Generation Pipeline")
    print("="*80)
    print(f"Configuration: {args.config}")
    print(f"  Protein: {config.protein}")
    print(f"  Input Data: {config.original_data_path}")
    print(f"  Checkpoints: {config.checkpoint_dir}")
    print(f"  Outputs: {config.output_dir}")
    if config.decode_to_coords:
        print(f"  Model: {config.model_path}")
        print(f"  PDB Reference: {config.pdb}")
        print(f"  PDB Samples: {config.n_pdb_samples}")
    print(f"  Training Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print("="*80)
    
    # 1. Train Autoencoder
    if not args.skip_ae:
        print("\n>>> Stage 1: Autoencoder Training")
        train_ae(config)
    else:
        print("\n>>> Skipping Autoencoder Training")

    # 2. Train DDPM
    if not args.skip_ddpm:
        print("\n>>> Stage 2: DDPM Training")
        train_ddpm(config)
    else:
        print("\n>>> Skipping DDPM Training")

    # 3. Generate & Evaluate
    print("\n>>> Stage 3: Generation & Evaluation")
    generate(config)
    
    # 4. Decode to coordinates and export PDBs
    if config.decode_to_coords:
        print("\n>>> Stage 4: Decoding to Coordinates and Exporting PDBs")
        decode_and_export_pdbs(config)
    
    print("\n" + "="*80)
    print("Pipeline Execution Completed Successfully!")
    print(f"Results saved to: {config.output_dir}")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - Generated samples: {config.output_dir}/generated_samples.npy")
    if config.decode_to_coords:
        print(f"  - PDB structures: {config.output_dir}/generated_pdbs/")
        print(f"  - MolProbity scores: {config.output_dir}/molprobity_scores.csv")
    print("="*80)


if __name__ == "__main__":
    main()

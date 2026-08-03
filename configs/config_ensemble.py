"""
Example configuration for ensemble generation with PDB export.
Copy this file and modify the parameters as needed.

Usage:
    from configs.config_ensemble import get_ensemble_config
    config = get_ensemble_config()
    
Or directly in ensemble_gen.py by modifying the Config initialization.
"""

from protscape.config import Config

def get_ensemble_config(protein_id="6e7e"):
    """
    Get configuration for ensemble generation with PDB export.
    
    Args:
        protein_id: Protein identifier (e.g., "7jfl", "7lp1", "ubiquitin")
    
    Returns:
        Config object with ensemble generation parameters
    """
    config = Config(
        # Data paths
        original_data_path="Inference/6e7e_A/latents_zrep.npy",
        checkpoint_dir=f"checkpoints/{protein_id}",
        output_dir=f"Generation/Ensemble/{protein_id}",
        
        # Decoding and PDB export
        decode_to_coords=True,
        protein=protein_id,
        model_path=f"train_logs/progsnn_logs_run_atlas_2026-04-08-052252/model_FINAL_6e7e_A.pt",
        dataset_path="data/graphs/",
        pdb="/nfs/roberts/pi/pi_sk2433/shared/ProtSCAPE_2026_MDSimulations/ATLAS_1K/6e7e_A/6e7e_A.pdb",  # Auto-detect from dataset
        n_pdb_samples=100,
        
        # XYZ normalization (if needed)
        normalize_xyz=False,
        xyz_mu_path=None,
        xyz_sd_path=None,
    )
    
    return config


# Example configurations for different proteins

def get_7jfl_config():
    """Configuration for 7jfl protein."""
    return get_ensemble_config(protein_id="7jfl")


def get_7lp1_config():
    """Configuration for 7lp1 protein."""
    return get_ensemble_config(protein_id="7lp1")


def get_ubiquitin_config():
    """Configuration for Ubiquitin."""
    config = get_ensemble_config(protein_id="ubiquitin")
    config.pdb = "data/datasets/Ubiquitin/ubiquitin.pdb"
    return config


def get_gb3_config():
    """Configuration for GB3."""
    config = get_ensemble_config(protein_id="gb3")
    config.pdb = "data/datasets/GB3/gb3.pdb"
    return config


if __name__ == "__main__":
    # Example: print configuration
    config = get_7jfl_config()
    print(config)

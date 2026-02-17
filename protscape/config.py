"""
Main configuration for the Hierarchical Latent Diffusion Model.
Can be initialized from YAML files for consistent configuration management.

Usage:
    # From YAML file:
    from utils.config import load_config
    config_dict = load_config("configs/config_ensemble.yaml")
    config = Config(**config_dict)
    
    # Direct instantiation:
    config = Config(protein="6h86", epochs=500)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import torch

@dataclass
class Config:
    """Best configuration: 16D Latent Space + Very Large DDPM."""
    
    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    input_dim: int = 128            # Original data dimension
    latent_dim: int = 16            # Compressed latent dimension
    data_path: str = "data/latents_compressed_16d.npy"
    original_data_path: str = "Ablations/7jfl/inference_test/7jfl_noatomAA/latents_zrep_10k.npy"
    
    # -------------------------------------------------------------------------
    # Autoencoder Configuration
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # DDPM Model Architecture (4.6M parameters)
    # -------------------------------------------------------------------------
    data_dim: int = 16              # Matches latent_dim
    hidden_dims: Optional[List[int]] = None        # Defaults to [512, 1024, 1024, 512]
    dropout: float = 0.1
    
    # -------------------------------------------------------------------------
    # Diffusion Parameters
    # -------------------------------------------------------------------------
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = 'cosine'
    clip_x0: float = 3.5            # Clipping range for generation
    
    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    batch_size: int = 256
    epochs: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_cosine_schedule: bool = True
    
    # -------------------------------------------------------------------------
    # EMA (Exponential Moving Average)
    # -------------------------------------------------------------------------
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # -------------------------------------------------------------------------
    # Paths & Logging
    # -------------------------------------------------------------------------
    save_every: int = 100
    
    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # -------------------------------------------------------------------------
    # Decoding & PDB Export Configuration
    # -------------------------------------------------------------------------
    decode_to_coords: bool = True
    protein: str = "7jfl_noatomAA"  # Used for naming outputs and loading specific models/datasets
    model_path: str = "train_logs/progsnn_logs_run_atlas_2026-02-11-130333/model_FINAL_7jfl_noatomAA.pt"
    dataset_path: str = "data/graphs/"
    pdb: Optional[str] = "data/datasets/7jfl_C_protein/7jfl_C.pdb"
    output_dir: str = "Generation/Ensemble/Ablations/7jfl_noatomAA"
    checkpoint_dir: str = "checkpoints/7jfl"
    ae_checkpoint: str = "checkpoints/7jfl/autoencoder_best.pt"
    ae_normalization: str = "checkpoints/7jfl/ae_normalization.npz"
    n_pdb_samples: int = 50
    normalize_xyz: bool = False
    xyz_mu_path: Optional[str] = None
    xyz_sd_path: Optional[str] = None

    # -------------------------------------------------------------------------
    feature_extractor: str = "scattering"  # Options: "scattering" (default), "gcn", "simple_gcn"
    # gcn_num_layers: int = 4         # Number of GCN layers (matches scattering depth)
    # gcn_hidden_channels: Optional[int] = None # null = same as input_dim
    
    # Node features - keep all enabled for fair comparison
    ablate_node_features: dict = field(default_factory=lambda: {
        "use_atomic_number": False,
        "use_residue_index": True,
        "use_amino_acid": False,
        "use_xyz": True,
        "randomize_atomic_number": False,
        "randomize_residue_index": False,
        "randomize_amino_acid": False,
        "randomize_xyz": False
    })
    
    # Edge features - keep enabled
    ablate_edge_features: dict = field(default_factory=lambda: {
        "use_edge_features": True,
        "randomize_edge_features": False,
        "zero_edge_features": False
    })
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 1024, 1024, 512]
        
        # Create directories
        Path(self.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        # Ensure AE checkpoint directory exists (e.g. checkpoints/)
        Path(self.ae_checkpoint).parent.mkdir(exist_ok=True, parents=True)
    
    def __str__(self):
        lines = ["Configuration:"]
        for key, value in self.__dict__.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

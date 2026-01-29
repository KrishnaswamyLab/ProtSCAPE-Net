"""
Main configuration for the Hierarchical Latent Diffusion Model.
"""
from dataclasses import dataclass
from pathlib import Path
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
    original_data_path: str = "Inference/6p5h/latents_zrep_10k.npy"
    
    # -------------------------------------------------------------------------
    # Autoencoder Configuration
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # DDPM Model Architecture (4.6M parameters)
    # -------------------------------------------------------------------------
    data_dim: int = 16              # Matches latent_dim
    hidden_dims: list = None        # Defaults to [512, 1024, 1024, 512]
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
    protein: str = "6p5h"
    model_path: str = "train_logs/progsnn_logs_run_deshaw_2026-01-27-140858/model_FINAL_6p5h.pt"
    dataset_path: str = "data/graphs/"
    pdb: str = "data/datasets/6p5h_A_protein/6p5h_A.pdb"
    output_dir: str = f"Generation/Ensemble/{protein}"
    checkpoint_dir: str = f"checkpoints/{protein}"
    ae_checkpoint: str = f"{checkpoint_dir}/autoencoder_best.pt"
    ae_normalization: str = f"{checkpoint_dir}/ae_normalization.npz"
    n_pdb_samples: int = 50
    normalize_xyz: bool = False
    xyz_mu_path: str = None
    xyz_sd_path: str = None
    
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

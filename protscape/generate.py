"""
Generation Script for Hierarchical Latent Diffusion Model.
Generates samples, computes metrics (MMD/Wasserstein), and creates static visualizations.
"""
import sys
# sys.path.insert(0, 'src') # Removed flattened structure

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import json

from protscape.config import Config
from protscape.model import DDPM
# from train import Normalizer  # Incorrect
from protscape.utils import Normalizer, compute_mmd, compute_wasserstein_1d_projections
from protscape.autoencoder import Autoencoder


def create_denoising_animation(config, num_samples=500):
    """Create 2D denoising animation (3D removed by request)."""
    print("\nCreating denoising animation (2D only)...")
    device = config.device
    
    # Load Models
    ddpm = DDPM(config)
    ddpm.load(Path(config.checkpoint_dir) / "best_model.pt")
    ddpm.model.eval()
    
    normalizer_16d = Normalizer()
    normalizer_16d.load(Path(config.checkpoint_dir) / "normalization.npz")
    
    ae = Autoencoder(input_dim=config.input_dim, latent_dim=config.latent_dim).to(device)
    ae.load(config.ae_checkpoint)
    ae.eval()
    
    ae_norm = np.load(config.ae_normalization)
    ae_mean, ae_std = ae_norm['mean'], ae_norm['std']
    
    # Load real data for reference
    real_data = np.load(config.original_data_path).astype(np.float32)
    
    # Prepare PCA
    pca_2d = PCA(n_components=2)
    real_pca_2d = pca_2d.fit_transform(real_data)
    
    # Denoising loop
    num_steps = ddpm.scheduler.num_timesteps
    plot_every = 20
    
    with torch.no_grad():
        # Start with pure noise
        x = torch.randn((num_samples, config.data_dim)).to(device)
        history_128d = []
        
        for i in range(num_steps):
            t = torch.full((num_samples,), num_steps - i - 1, device=device, dtype=torch.long)
            
            # Denoising step (using correct sample_step method)
            x = ddpm.scheduler.sample_step(ddpm.model, x, t, num_steps - i - 1)
            
            if (num_steps - i - 1) % plot_every == 0 or i == num_steps - 1:
                # Denormalize 16D samples
                x_denorm = normalizer_16d.inverse_transform(x.cpu().numpy())
                # Decode to 128D
                z_tensor = torch.from_numpy(x_denorm).to(device)
                decoded_norm = ae.decode(z_tensor).cpu().numpy()
                # Denormalize 128D
                gen_128d = decoded_norm * ae_std + ae_mean
                history_128d.append(gen_128d)
    
    # Create 2D Animation
    print("Saving 2D animation...")
    fig_2d, ax_2d = plt.subplots(figsize=(8, 8))
    ax_2d.scatter(real_pca_2d[:, 0], real_pca_2d[:, 1], alpha=0.1, s=1, c='blue', label='Real')
    scatter_2d = ax_2d.scatter([], [], alpha=0.5, s=2, c='red', label='Generating...')
    ax_2d.set_title("Denoising Process (PCA 2D)")
    ax_2d.legend()
    
    def update_2d(frame):
        gen_pca = pca_2d.transform(history_128d[frame])
        scatter_2d.set_offsets(gen_pca)
        return scatter_2d,

    ani_2d = animation.FuncAnimation(fig_2d, update_2d, frames=len(history_128d), interval=100)
    ani_2d.save(Path(config.output_dir) / "denoising_2d.gif", writer='pillow')
    plt.close()
    
    # 3D Animation removed per request


def generate(config=None, num_samples=10000):
    """Main generation function."""
    if config is None:
        config = Config()
    print("="*60)
    print("Generation & Evaluation")
    print("="*60)
    
    device = config.device
    
    # Load Models
    print("Loading models...")
    ddpm = DDPM(config)
    ddpm.load(Path(config.checkpoint_dir) / "best_model.pt")
    
    normalizer_16d = Normalizer()
    normalizer_16d.load(Path(config.checkpoint_dir) / "normalization.npz")
    
    ae = Autoencoder(input_dim=config.input_dim, latent_dim=config.latent_dim).to(device)
    ae.load(config.ae_checkpoint)
    ae.eval()
    
    ae_norm = np.load(config.ae_normalization)
    ae_mean, ae_std = ae_norm['mean'], ae_norm['std']
    
    # Generate
    print(f"\nGenerating {num_samples} samples...")
    ddpm.model.eval()
    with torch.no_grad():
        compressed_norm = ddpm.sample(num_samples)
        compressed_norm = compressed_norm.cpu().numpy()
    
    # Decode
    print("Decoding to original space...")
    compressed = normalizer_16d.inverse_transform(compressed_norm)
    with torch.no_grad():
        z_tensor = torch.from_numpy(compressed).to(device)
        print(f"Decoding {z_tensor.shape}...") # Debug print
        decoded_norm = ae.decode(z_tensor).cpu().numpy()
    
    generated_data = decoded_norm * ae_std + ae_mean
    
    # Save
    save_path = Path(config.output_dir) / "generated_samples.npy"
    np.save(save_path, generated_data)
    print(f"Saved samples to {save_path}")
    
    # Evaluation
    print("Computing metrics...")
    real_data = np.load(config.original_data_path).astype(np.float32)
    
    # Compute metrics
    # Random sample for MMD to avoid memory issues if dataset is massive
    # but 10k is fine for RBF kernel usually (~1GB-ish matrix)
    # Sample from min length to match sizes
    min_len = min(len(real_data), len(generated_data))
    indices = np.random.choice(min_len, min(min_len, 5000), replace=False)
    
    real_sample_mmd = real_data[indices]
    gen_sample_mmd = generated_data[indices]
    
    mmd = compute_mmd(real_sample_mmd, gen_sample_mmd)
    wasserstein = compute_wasserstein_1d_projections(real_data, generated_data)
    
    print(f"\nFinal Metrics:")
    print(f"  MMD (RBF kernel): {mmd:.6f}")
    print(f"  Wasserstein distance: {wasserstein:.6f}")
    print(f"  Generated 128D std: {generated_data.std():.4f} (Real: {real_data.std():.4f})")
    
    # Save metrics to JSON
    metrics = {
        "mmd": float(mmd),
        "wasserstein": float(wasserstein),
        "std_generated": float(generated_data.std()),
        "std_real": float(real_data.std())
    }
    with open(Path(config.output_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Visualizations
    print("Creating visualizations...")
    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_data)
    gen_pca = pca.transform(generated_data)
    
    # 2D Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.3, s=1, c='blue')
    axes[0].set_title('Real Data')
    axes[1].scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.3, s=1, c='red')
    axes[1].set_title('Generated Data')
    axes[2].scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.2, s=1, c='blue', label='Real')
    axes[2].scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.2, s=1, c='red', label='Generated')
    axes[2].set_title('Overlay')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(Path(config.output_dir) / "pca_comparison_2d.png", dpi=150)
    plt.close()
    
    # 3D Comparison (Static)
    print("Creating 3D static plots...")
    pca_3d = PCA(n_components=3)
    real_pca_3d = pca_3d.fit_transform(real_data)
    gen_pca_3d = pca_3d.transform(generated_data)
    
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(real_pca_3d[:, 0], real_pca_3d[:, 1], real_pca_3d[:, 2], alpha=0.3, s=1, c='blue')
    ax1.set_title('Real Data (3D)')
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(gen_pca_3d[:, 0], gen_pca_3d[:, 1], gen_pca_3d[:, 2], alpha=0.3, s=1, c='red')
    ax2.set_title('Generated Data (3D)')
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(real_pca_3d[:, 0], real_pca_3d[:, 1], real_pca_3d[:, 2], alpha=0.2, s=1, c='blue', label='Real')
    ax3.scatter(gen_pca_3d[:, 0], gen_pca_3d[:, 1], gen_pca_3d[:, 2], alpha=0.2, s=1, c='red', label='Generated')
    ax3.set_title('Overlay (3D)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(Path(config.output_dir) / "pca_comparison_3d.png", dpi=150)
    plt.close()
    
    print("\nVisualizations saved to:", config.output_dir)
    
    # Generate animations (2D only)
    create_denoising_animation(config)
    
    print("\nGeneration completed!")


if __name__ == "__main__":
    generate()

"""
Autoencoder Training Script.
Trains an MLP Autoencoder to compress protein latents from 128D to 16D.
"""
import sys
# sys.path.insert(0, 'src') # Removed flattened structure

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from protscape.config import Config
from protscape.autoencoder import Autoencoder, count_parameters


def train_ae(config=None):
    """Train autoencoder using configuration."""
    if config is None:
        config = Config()
    
    print("="*60)
    print(f"Training Autoencoder: {config.input_dim}D → {config.latent_dim}D → {config.input_dim}D")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {config.original_data_path}...")
    if not Path(config.original_data_path).exists():
        raise FileNotFoundError(f"Data file not found at {config.original_data_path}")
        
    data = np.load(config.original_data_path).astype(np.float32)
    print(f"Data shape: {data.shape}")
    
    # Normalize data
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    data_norm = (data - mean) / std
    print(f"Normalized range: [{data_norm.min():.4f}, {data_norm.max():.4f}]")
    
    # Save normalization stats
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    # Using the path from config, but config points to ae_normalization which is in checkpoints/
    # Let's ensure we save it where config expects it
    np.savez(config.ae_normalization, mean=mean, std=std)
    print(f"Saved normalization stats to {config.ae_normalization}")
    
    # Create dataloader
    dataset = TensorDataset(torch.from_numpy(data_norm))
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = Autoencoder(
        input_dim=config.input_dim, 
        latent_dim=config.latent_dim
    ).to(config.device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Optimizer
    # Note: AE training might use different LR than DDPM, using default 1e-3 here per previous successful runs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Use 200 epochs for AE as established in previous experiments
    ae_epochs = 200 
    print(f"Training for {ae_epochs} epochs...")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ae_epochs, eta_min=1e-5)
    criterion = nn.MSELoss()
    
    # Training results
    losses = []
    best_loss = float('inf')
    
    # Training Loop
    model.train()
    print("\nStarting training...")
    for epoch in range(ae_epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{ae_epochs}")
        for batch in pbar:
            x = batch[0].to(config.device)
            
            # Forward
            x_recon, z = model(x)
            loss = criterion(x_recon, x)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # Scheduler step
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            lr_current = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{ae_epochs} - Loss: {avg_loss:.6f} - LR: {lr_current:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save(config.ae_checkpoint)
            
    # Save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.title('Autoencoder Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(config.output_dir) / "ae_loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print("Autoencoder training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Model saved to: {config.ae_checkpoint}")
    print("="*60)
    
    # Extract latents
    print("\nExtracting compressed latents...")
    model.eval()
    all_latents = []
    
    # Use data loader without shuffling for consistent ordering if needed, 
    # but technically for checking distribution we usually want the whole dataset.
    # To save the compressed dataset for DDPM training, we should pass the whole dataset
    # in order.
    
    full_dataset_loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
    )
    
    with torch.no_grad():
        for batch in tqdm(full_dataset_loader, desc="Encoding"):
            x = batch[0].to(config.device)
            z = model.encode(x)
            all_latents.append(z.cpu().numpy())
            
    latents = np.concatenate(all_latents, axis=0)
    print(f"Extracted latents shape: {latents.shape}")
    
    np.save(config.data_path, latents)
    print(f"Saved compressed latents to: {config.data_path}")


if __name__ == "__main__":
    train_ae()

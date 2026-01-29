"""
Main Training Script for the Hierarchical Latent Diffusion Model (Stage 2).
Trains the DDPM on compressed 16D latents.
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
from protscape.model import DDPM
from protscape.utils import EMA, Normalizer


def load_and_prepare_data(config):
    """Load data and apply normalization."""
    # Load data
    print(f"Loading data from {config.data_path}...")
    if not Path(config.data_path).exists():
        raise FileNotFoundError(f"Data file not found at {config.data_path}. Please run latent extraction first.")
        
    data = np.load(config.data_path).astype(np.float32)
    print(f"Data shape: {data.shape}")
    
    # Normalize
    normalizer = Normalizer()
    normalizer.fit(data)
    data_normalized = normalizer.transform(data)
    
    print(f"Normalized data mean: {data_normalized.mean():.4f}, std: {data_normalized.std():.4f}")
    
    # Save normalization stats
    normalizer.save(Path(config.checkpoint_dir) / "normalization.npz")
    
    # Create dataset and dataloader
    dataset = TensorDataset(torch.from_numpy(data_normalized))
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    return dataloader, normalizer


def train_ddpm(config=None):
    """Main training loop."""
    if config is None:
        config = Config()
    
    print("="*60)
    print("DDPM Training (Hierarchical 16D)")
    print("="*60)
    print(config)
    print("="*60)
    
    # Load data
    dataloader, normalizer = load_and_prepare_data(config)
    
    # Create model
    ddpm = DDPM(config)
    
    # Create EMA if enabled
    ema = None
    if config.use_ema:
        ema = EMA(ddpm.model, decay=config.ema_decay)
        print(f"EMA enabled with decay={config.ema_decay}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        ddpm.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler
    if config.use_cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01,
        )
    else:
        scheduler = None
    
    # Training loop
    ddpm.model.train()
    losses = []
    best_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(config.epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            x_batch = batch[0].to(ddpm.device)
            
            # Forward pass
            loss = ddpm.train_step(x_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), config.grad_clip)
            
            optimizer.step()
            
            # Update EMA
            if ema is not None:
                ema.update()
            
            # Log
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        lr = optimizer.param_groups[0]['lr']
        # Print less frequently to avoid clutter
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.epochs} - Loss: {avg_loss:.6f} - LR: {lr:.6f}")
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save checkpoint (periodically)
        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            if ema is not None:
                ema.apply_shadow()
                ddpm.save(checkpoint_path)
                ema.restore()
            else:
                ddpm.save(checkpoint_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = Path(config.checkpoint_dir) / "best_model.pt"
            if ema is not None:
                ema.apply_shadow()  # Use EMA weights for best model
                ddpm.save(best_path)
                ema.restore()  # Restore training weights
            else:
                ddpm.save(best_path)
    
    # Save final model
    final_path = Path(config.checkpoint_dir) / "final_model.pt"
    if ema is not None:
        ema.apply_shadow()
        ddpm.save(final_path)
        ema.restore()
    else:
        ddpm.save(final_path)
    
    # Save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(config.output_dir) / "loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Best model saved to: {best_path}")
    print("="*60)


if __name__ == "__main__":
    train_ddpm()

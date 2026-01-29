"""
DDPM model for point cloud generation.
Includes MLP-based denoiser and noise scheduler.
"""
import torch
import torch.nn as nn
import numpy as np
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with layer norm and dropout."""
    
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return x + self.block(x)


class MLPDenoiser(nn.Module):
    """MLP-based denoising model with residual connections."""
    
    def __init__(self, data_dim, hidden_dims, dropout=0.1, time_emb_dim=64):
        super().__init__()
        self.data_dim = data_dim
        self.time_emb_dim = time_emb_dim
        
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        
        # Input projection
        self.input_proj = nn.Linear(data_dim + time_emb_dim, hidden_dims[0])
        
        # Hidden layers with residual connections
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            # Add residual block if dimensions match
            if hidden_dims[i] == hidden_dims[i + 1]:
                layers.append(ResidualBlock(hidden_dims[i + 1], dropout))
        
        self.net = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dims[-1], data_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x, t):
        """
        Args:
            x: (batch, data_dim) - noisy data
            t: (batch,) - timestep indices
        Returns:
            (batch, data_dim) - predicted noise
        """
        # Embed timestep
        t_emb = self.time_mlp(t)
        
        # Concatenate data and time embedding
        x = torch.cat([x, t_emb], dim=-1)
        
        # Process through network
        x = self.input_proj(x)
        x = self.net(x)
        x = self.output_proj(x)
        
        return x


class DDPMScheduler:
    """DDPM noise scheduler for forward and reverse diffusion."""
    
    def __init__(self, num_timesteps, beta_start, beta_end, schedule='cosine', device='cuda', clip_x0=3.0):
        self.num_timesteps = num_timesteps
        self.device = device
        self.clip_x0 = clip_x0  # Store clipping value
        self.beta_schedule = schedule # Changed from beta_schedule to schedule
        
        # Beta schedule
        if self.beta_schedule == 'cosine': # Changed from beta_schedule to self.beta_schedule
            # Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models"
            steps = num_timesteps + 1 # Changed from timesteps to num_timesteps
            s = 0.008  # offset
            x = torch.linspace(0, num_timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(self.betas, 0.0001, 0.9999)  # Clip for numerical stability
        else:
            # Linear beta schedule
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        
        # For forward diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For reverse diffusion (sampling)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_start, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0)
        Args:
            x_start: (batch, dim) - clean data
            t: (batch,) - timestep indices
            noise: (batch, dim) - optional noise, sampled if None
        Returns:
            noisy_x: (batch, dim) - noisy data
            noise: (batch, dim) - the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        noisy_x = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
        return noisy_x, noise
    
    @torch.no_grad()
    def sample_step(self, model, x, t, t_index):
        """
        Single reverse diffusion step: p(x_{t-1} | x_t)
        Args:
            model: denoising model
            x: (batch, dim) - data at timestep t
            t: (batch,) - timestep indices
            t_index: int - current timestep index
        Returns:
            (batch, dim) - data at timestep t-1
        """
        # Predict noise
        predicted_noise = model(x, t)
        
        # Get coefficients
        alpha_cumprod = self.alphas_cumprod[t][:, None]
        alpha_cumprod_prev = self.alphas_cumprod_prev[t][:, None]
        beta = self.betas[t][:, None]
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        # Predict x_0 from x_t and predicted noise
        pred_x0 = (x - sqrt_one_minus_alpha_cumprod *predicted_noise) / sqrt_alpha_cumprod
        
        # Clip predicted x_0 to prevent variance explosion (configurable)
        pred_x0 = torch.clamp(pred_x0, -self.clip_x0, self.clip_x0)
        
        # Compute coefficients for x_{t-1}
        # Use DDPM posterior mean: mu = (sqrt(alpha_prev) * beta_t) / (1 - alpha_bar_t) * x_0 + (sqrt(alpha_t) * (1 - alpha_bar_{t-1})) / (1 - alpha_bar_t) * x_t
        coef_x0 = torch.sqrt(alpha_cumprod_prev) * beta / (1.0 - alpha_cumprod)
        coef_xt = torch.sqrt(self.alphas[t][:, None]) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        
        pred_mean = coef_x0 * pred_x0 + coef_xt * x
        
        if t_index == 0:
            # No noise at final step
            return pred_mean
        else:
            # Add noise scaled by posterior variance
            noise = torch.randn_like(x)
            variance = self.posterior_variance[t][:, None]
            # Clip variance to prevent numerical issues
            variance = torch.clamp(variance, min=1e-20)
            return pred_mean + torch.sqrt(variance) * noise
    
    @torch.no_grad()
    def sample(self, model, shape, return_trajectory=False):
        """
        Full reverse diffusion process.
        Args:
            model: denoising model
            shape: tuple - shape of samples to generate
            return_trajectory: bool - if True, return all intermediate steps
        Returns:
            samples: (batch, dim) - generated samples
            trajectory: list of (batch, dim) - intermediate steps (if return_trajectory=True)
        """
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        trajectory = [x.cpu().numpy()] if return_trajectory else None
        
        # Reverse diffusion
        for t_index in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), t_index, device=self.device, dtype=torch.long)
            x = self.sample_step(model, x, t, t_index)
            
            if return_trajectory and (t_index % 50 == 0 or t_index == 0):
                trajectory.append(x.cpu().numpy())
        
        if return_trajectory:
            return x, trajectory
        return x


class DDPM:
    """Wrapper class combining model and scheduler."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.clip_x0 = config.clip_x0  # Configurable clipping
        
        # Create model
        self.model = MLPDenoiser(
            data_dim=config.data_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(self.device)
        
        # Create scheduler
        self.scheduler = DDPMScheduler(
            num_timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule=config.beta_schedule,
            device=self.device,
            clip_x0=self.clip_x0,  # Pass clipping value
        )
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Device: {self.device}")
    
    def train_step(self, x_batch):
        """Single training step."""
        # Sample random timesteps
        t = torch.randint(0, self.config.timesteps, (x_batch.shape[0],), device=self.device)
        
        # Add noise
        noisy_x, noise = self.scheduler.add_noise(x_batch, t)
        
        # Predict noise
        predicted_noise = self.model(noisy_x, t)
        
        # MSE loss
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(self, num_samples, return_trajectory=False):
        """Generate samples."""
        self.model.eval()
        shape = (num_samples, self.config.data_dim)
        
        if return_trajectory:
            samples, trajectory = self.scheduler.sample(self.model, shape, return_trajectory=True)
            return samples, trajectory
        else:
            samples = self.scheduler.sample(self.model, shape, return_trajectory=False)
            return samples
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {path}")

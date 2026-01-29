"""
Autoencoder to compress 128D latents to lower dimensional space.
Then run diffusion in the compressed space.
"""
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """MLP autoencoder to compress latent space."""
    
    def __init__(self, input_dim=128, latent_dim=32, hidden_dims=[256, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: input_dim -> latent_dim
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: latent_dim -> input_dim
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode to compressed latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from compressed latent space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass: encode then decode."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
        }, path)
        print(f"Autoencoder saved to {path}")
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Autoencoder loaded from {path}")
        return self


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""
Helper utilities for training and data processing.
"""
import numpy as np
import torch
from copy import deepcopy
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel


class EMA:
    """
    Exponential Moving Average for model parameters.
    Standard practice in diffusion models for stable sampling.
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Replace model parameters with EMA shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Get state dict for checkpointing."""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


class Normalizer:
    """Standardize data (z-score normalization) and reverse."""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        """Compute mean and std from data."""
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        # Avoid division by zero
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self
    
    def transform(self, data):
        """Normalize data."""
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        """Denormalize data."""
        return data * self.std + self.mean
    
    def save(self, path):
        """Save normalization parameters."""
        np.savez(path, mean=self.mean, std=self.std)
        print(f"Normalization stats saved to {path}")
    
    def load(self, path):
        """Load normalization parameters."""
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']
        print(f"Normalization stats loaded from {path}")
        return self


def compute_mmd(x, y, kernel='rbf'):
    """Compute Maximum Mean Discrepancy (MMD) between two sets of samples."""
    x_kernel = rbf_kernel(x, x)
    y_kernel = rbf_kernel(y, y)
    xy_kernel = rbf_kernel(x, y)
    return np.mean(x_kernel) + np.mean(y_kernel) - 2 * np.mean(xy_kernel)


def compute_wasserstein_1d_projections(x, y, num_projections=1000):
    """
    Compute Sliced Wasserstein Distance by projecting to 1D lines.
    Approximates the Wasserstein distance between high-dimensional distributions.
    """
    dim = x.shape[1]
    projections = np.random.normal(0, 1, (dim, num_projections))
    projections /= np.linalg.norm(projections, axis=0)

    x_proj = x @ projections
    y_proj = y @ projections

    wasserstein_dists = []
    for i in range(num_projections):
        w_dist = wasserstein_distance(x_proj[:, i], y_proj[:, i])
        wasserstein_dists.append(w_dist)

    return np.mean(wasserstein_dists)

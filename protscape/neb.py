"""
Nudged Elastic Band (NEB) implementation for minimum energy path finding.

This module provides functions for optimizing paths in latent space using the
Nudged Elastic Band method, which finds minimum energy pathways between 
conformational states.

References:
    Henkelman, G., & Jónsson, H. (2000). 
    Improved tangent estimate in the nudged elastic band method for finding 
    minimum energy paths and saddle points. J. Chem. Phys., 113(22), 9978-9985.
"""

from typing import Callable, List
import numpy as np
import torch


def redistribute_pivots(pivots):
    """
    Redistribute pivots along the path to equal arc length spacing.
    
    This maintains equal spacing between consecutive pivots along the path,
    which is important for NEB convergence and prevents clustering.
    
    Args:
        pivots: List of numpy arrays representing pivot positions
        
    Returns:
        List of redistributed pivot positions with equal spacing
    """
    pivots = [np.asarray(p) for p in pivots]
    num_pivots = len(pivots)
    if num_pivots < 2:
        return pivots

    # Compute segment lengths
    segment_lengths = [np.linalg.norm(pivots[i+1] - pivots[i]) 
                      for i in range(num_pivots - 1)]
    total_path_length = sum(segment_lengths)
    if total_path_length == 0:
        return pivots

    # Compute cumulative arc length
    cumulative_lengths = np.zeros(num_pivots)
    cumulative_lengths[1:] = np.cumsum(segment_lengths)
    
    # Target equally-spaced arc lengths
    new_distances = np.linspace(0, total_path_length, num_pivots)

    # Interpolate new pivot positions
    new_pivots = [pivots[0].copy()]
    for i in range(1, num_pivots - 1):
        target_dist = new_distances[i]
        segment_idx = np.searchsorted(cumulative_lengths, target_dist, side='right') - 1
        if segment_idx < 0:
            segment_idx = 0
        segment_start_dist = cumulative_lengths[segment_idx]
        segment_len = segment_lengths[segment_idx]
        alpha = 0.0 if segment_len == 0 else (target_dist - segment_start_dist) / segment_len
        p_start = pivots[segment_idx]
        p_end = pivots[segment_idx + 1]
        p_new = p_start + alpha * (p_end - p_start)
        new_pivots.append(p_new)

    new_pivots.append(pivots[-1].copy())
    return new_pivots


def get_tangent(p_prev, p_i, p_next, L_prev, L_next):
    """
    Compute tangent vector for NEB using improved tangent estimate.
    
    The tangent direction is chosen based on adjacent energy values to
    handle kinks and corners in the energy landscape properly.
    
    Args:
        p_prev: Previous pivot position
        p_i: Current pivot position
        p_next: Next pivot position
        L_prev: Energy at previous pivot
        L_next: Energy at next pivot
        
    Returns:
        Normalized tangent vector
    """
    tau = (p_next - p_i) if (L_next > L_prev) else (p_i - p_prev)
    n = np.linalg.norm(tau)
    return np.zeros_like(tau) if n == 0 else tau / n


def neb_step(pivots, get_loss_and_grad_func, learning_rate):
    """
    Perform one NEB optimization step with stabilized updates.
    
    This implements the string method variant of NEB:
    1. Redistribute pivots to equal arc length
    2. Compute energy and gradients at all pivots
    3. Project gradient perpendicular to path tangent
    4. Update pivots with normalized step (fixed magnitude)
    
    Args:
        pivots: List of current pivot positions
        get_loss_and_grad_func: Function that returns (energy, gradient) for a position
        learning_rate: Step size for updates (actual distance moved)
        
    Returns:
        List of updated pivot positions
    """
    current_pivots = redistribute_pivots(pivots)
    num_internal_pivots = len(current_pivots) - 2

    # Compute energies and gradients at all pivots
    losses_and_grads = [get_loss_and_grad_func(p) for p in current_pivots]
    new_pivots = [current_pivots[0].copy()]

    # Update internal pivots (keep endpoints fixed)
    for i in range(1, num_internal_pivots + 1):
        p_prev, p_i, p_next = current_pivots[i-1], current_pivots[i], current_pivots[i+1]
        L_prev, _ = losses_and_grads[i-1]
        L_next, _ = losses_and_grads[i+1]
        _, grad_L_i = losses_and_grads[i]

        # Compute tangent direction
        tau_i = get_tangent(p_prev, p_i, p_next, L_prev, L_next)

        # Project gradient perpendicular to tangent
        grad_parallel = np.dot(grad_L_i, tau_i) * tau_i
        grad_perpendicular = grad_L_i - grad_parallel

        # Stabilized step: normalize direction, fixed step length
        step_dir = -grad_perpendicular
        norm = np.linalg.norm(step_dir) + 1e-12
        step = (learning_rate * step_dir) / norm

        p_new = p_i + step
        new_pivots.append(p_new)

    new_pivots.append(current_pivots[-1].copy())
    return new_pivots


def build_neb_loss_and_grad_fn(model, grad_fn, device):
    """
    Build a loss and gradient function for NEB in latent space.
    
    This wraps the existing gradient function (from OpenMM forces) to work
    with NEB's numpy-based optimization.
    
    Args:
        model: Neural network model for decoding latent to coordinates
        grad_fn: Existing gradient function (computes OpenMM forces)
        device: torch device
        
    Returns:
        Function that takes numpy array and returns (energy, gradient)
    """
    def z_loss_and_grad(z_np):
        """
        Compute energy and gradient in latent space.
        
        Args:
            z_np: Latent position (numpy array)
            
        Returns:
            (energy, gradient): Scalar energy and gradient vector
        """
        z_t = torch.tensor(z_np, dtype=torch.float, device=device).unsqueeze(0).requires_grad_(True)
        
        # Get gradient in latent space from OpenMM forces
        grad_z_t = grad_fn(z_t)  # (1, D)
        
        # Compute energy as norm of gradient (proxy for potential energy)
        # This is a heuristic; ideally we'd evaluate actual potential energy
        energy = float(torch.norm(grad_z_t).item())
        
        # Return gradient for chain rule
        grad_z_np = grad_z_t.detach().cpu().numpy().squeeze(0)
        
        return energy, grad_z_np
    
    return z_loss_and_grad


def generate_neb_paths(
    grad_fn: Callable,
    latents_all: torch.Tensor,
    start_indices: List[int],
    args,
    device: torch.device,
    model,
) -> torch.Tensor:
    """
    Generate multiple NEB paths in parallel.
    
    For each starting point, this function:
    1. Initializes a path via linear interpolation to a target
    2. Optimizes the path using NEB to find minimum energy pathway
    3. Returns the optimized path as a trajectory
    
    Args:
        grad_fn: Gradient function in latent space
        latents_all: All latent embeddings (N, D)
        start_indices: Starting point indices for each path
        args: Configuration arguments (must contain n_pivots, neb_steps, neb_lr)
        device: torch device
        model: Neural network model
        
    Returns:
        traj: Trajectory tensor of shape (T, B, D) where T=n_pivots+1, B=n_paths
    """
    n_paths = len(start_indices)
    n_pivots = getattr(args, 'n_pivots', 20)
    neb_steps = getattr(args, 'neb_steps', 50)
    neb_lr = getattr(args, 'neb_lr', 0.05)
    
    # Build NEB loss/grad function
    z_loss_grad_fn = build_neb_loss_and_grad_fn(model, grad_fn, device)
    
    # Find global minimum energy point as common target
    # (or use a predefined target if specified)
    lat_np = latents_all.detach().cpu().numpy()
    
    # Simple heuristic: use the point with lowest gradient norm as target
    # Or use args.end_idx if specified
    if hasattr(args, 'end_idx') and args.end_idx is not None:
        end_idx = args.end_idx
    else:
        # Find point with minimum gradient norm (sample subset for efficiency)
        print("[NEB] Auto-detecting target endpoint...")
        grad_norms = []
        sample_size = min(1000, len(lat_np))
        for i in range(sample_size):
            _, grad = z_loss_grad_fn(lat_np[i])
            grad_norms.append(np.linalg.norm(grad))
        end_idx = int(np.argmin(grad_norms))
    
    print(f"[NEB] Using end_idx={end_idx} as target for all paths")
    
    z_end = lat_np[end_idx].astype(float)
    
    all_paths = []
    
    for path_idx, start_idx in enumerate(start_indices):
        print(f"[NEB] Generating path {path_idx+1}/{n_paths} from idx={start_idx} to idx={end_idx}")
        
        z_start = lat_np[start_idx].astype(float)
        
        # Initialize linear interpolation between start and end
        initial_pivots = [z_start + (z_end - z_start) * (i / float(n_pivots)) 
                         for i in range(n_pivots + 1)]
        
        # Run NEB optimization
        path_pivots = initial_pivots.copy()
        for step in range(neb_steps):
            path_pivots = neb_step(path_pivots, z_loss_grad_fn, learning_rate=neb_lr)
            
            if (step + 1) % 10 == 0:
                # Compute path energies for monitoring
                energies = [z_loss_grad_fn(p)[0] for p in path_pivots]
                max_e = max(energies)
                mean_e = np.mean(energies)
                print(f"  [NEB step {step+1}/{neb_steps}] max_energy={max_e:.4e}, mean_energy={mean_e:.4e}")
        
        # Store final path
        final_path = np.array(path_pivots)  # (n_pivots+1, D)
        all_paths.append(final_path)
    
    # Stack all paths: (B, T, D) -> transpose to (T, B, D)
    all_paths_np = np.stack(all_paths, axis=0)  # (B, T, D)
    all_paths_np = np.transpose(all_paths_np, (1, 0, 2))  # (T, B, D)
    
    # Convert to torch tensor
    traj = torch.tensor(all_paths_np, dtype=torch.float32, device=device)
    
    return traj

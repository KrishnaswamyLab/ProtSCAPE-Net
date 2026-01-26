"""
Metropolis-Adjusted Langevin Algorithm (MALA) for sampling in latent space.

This module implements MALA with volume correction for sampling from Boltzmann
distributions in ambient space while performing dynamics in the latent space
of an autoencoder.

Key features:
  - Volume correction via decoder Jacobian
  - Support for cycle-consistency manifold penalty
  - Batched sampling with per-sample accept/reject

References:
  - Girolami & Calderhead (2011) "Riemann manifold Langevin and Hamiltonian..."
  - Zhu et al. (2017) "Unpaired Image-to-Image Translation using Cycle-Consistent..."

Author: Xingzhi Sun
Date: 2026-01-24
"""

import torch
from tqdm import tqdm
from typing import Callable, Optional


def mfd_mala(
    f_fn: Callable[[torch.Tensor], torch.Tensor],
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    log_vol_fn: Callable[[torch.Tensor], torch.Tensor],
    x_init: torch.Tensor,
    num_steps: int,
    step_size: float,
    step_size_noise: Optional[float] = None,
    return_history: bool = False,
    stop_func: Optional[Callable[[torch.Tensor], bool]] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    MALA (Metropolis-Adjusted Langevin Algorithm) in latent space with volume correction.
    
    Samples from Boltzmann distribution p(x) ∝ exp(-U(x)) in ambient space by running 
    MALA in latent space z with volume correction for the decoder Jacobian.
    
    IMPORTANT: The energy function should include a cycle-consistency penalty to prevent
    samples from drifting to out-of-distribution regions of latent space:
        U_total(z) = U_physical(decoder(z)) + λ * ||encoder(decoder(z)) - z||²
    
    The algorithm performs:
      1. Propose: z' = z + dt * score(z) + sqrt(2*dt) * noise
      2. Compute acceptance ratio with volume correction
      3. Accept/reject via Metropolis-Hastings
    
    Args:
        f_fn: Function that returns grad_z U (gradient of energy w.r.t. latent z).
              Input: [batch, latent_dim], Output: [batch, latent_dim]
        energy_fn: Function that returns U(z) (total energy including manifold penalty).
              Input: [batch, latent_dim], Output: [batch]
        log_vol_fn: Function that returns log volume element = 0.5 * logdet(J^T J),
              where J is the decoder Jacobian.
              Input: [batch, latent_dim], Output: [batch]
        x_init: Initial latent samples. Shape: [batch, latent_dim]
        num_steps: Number of MALA steps to perform.
        step_size: Step size for drift term (dt in the equation above).
        step_size_noise: Step size for noise term. If None, uses step_size (assumes k_B*T=1).
        return_history: If True, return all intermediate samples instead of just final.
        stop_func: Optional early stopping function. Called with current samples,
                  should return True to stop.
        seed: Random seed for reproducibility.
    
    Returns:
        torch.Tensor: 
            - If return_history=False: Final samples [batch, latent_dim]
            - If return_history=True: All samples [num_steps, batch, latent_dim]
    
    Example:
        >>> # Setup functions (example, not runnable)
        >>> energy_fn = lambda z: U_physical(decoder(z)) + penalty * ||encoder(decoder(z)) - z||²
        >>> force_fn = lambda z: autograd.grad(energy_fn(z).sum(), z)[0]
        >>> log_vol_fn = lambda z: 0.5 * logdet(jacobian(decoder, z).T @ jacobian(decoder, z))
        >>> 
        >>> # Run MALA
        >>> samples = mfd_mala(
        ...     f_fn=force_fn,
        ...     energy_fn=energy_fn,
        ...     log_vol_fn=log_vol_fn,
        ...     x_init=initial_latents,
        ...     num_steps=5000,
        ...     step_size=0.002,
        ... )
    
    References:
        - Volume correction: Girolami & Calderhead (2011) "Riemann manifold Langevin 
          and Hamiltonian Monte Carlo methods", J. R. Stat. Soc. B, 73(2):123-214.
        - Manifold penalty (cycle consistency): Zhu et al. (2017) "Unpaired Image-to-Image 
          Translation using Cycle-Consistent Adversarial Networks", ICCV 2017.
    
    Notes:
        - Assumes temperature k_B*T = 1 (can be absorbed into energy function)
        - Uses detach() to prevent computation graph from growing
        - Each sample in batch has independent accept/reject decision
        - Typical acceptance rates: 50-70% (adjust step_size if too high/low)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if step_size_noise is None:
        step_size_noise = step_size
    
    # Initialize state
    x = x_init.clone().detach().requires_grad_(True)
    
    # Compute initial quantities
    grad_f = f_fn(x)
    score = -grad_f
    
    current_energy = energy_fn(x)
    current_log_vol = log_vol_fn(x)

    # Storage for history
    if return_history:
        history = []

    # Main MALA loop
    for step in tqdm(range(num_steps), desc="MALA sampling"):
        # =====================================================================
        # 1. Propose new state
        # =====================================================================
        noise = torch.randn_like(x)
        sqrt_step_size = (2 * step_size_noise) ** 0.5
        
        noise_term = sqrt_step_size * noise
        drift_term = step_size * score
        x_prop = x + drift_term + noise_term

        # =====================================================================
        # 2. Compute proposal quantities
        # =====================================================================
        x_prop = x_prop.detach().requires_grad_(True)
        grad_f_prop = f_fn(x_prop)
        score_prop = -grad_f_prop
        
        prop_energy = energy_fn(x_prop)
        prop_log_vol = log_vol_fn(x_prop)

        # =====================================================================
        # 3. MALA acceptance ratio
        # =====================================================================
        # Forward proposal probability: q(x_prop | x)
        fwd_diff = x_prop - (x + drift_term)
        log_q_fwd = -0.5 * torch.sum(fwd_diff**2, dim=1) / (2 * step_size_noise)

        # Reverse proposal probability: q(x | x_prop)
        rev_drift = step_size * score_prop
        rev_diff = x - (x_prop + rev_drift)
        log_q_rev = -0.5 * torch.sum(rev_diff**2, dim=1) / (2 * step_size_noise)

        # Acceptance ratio components:
        #   - Energy change: -ΔU (lower energy = more likely to accept)
        #   - Volume change: Δlog_vol (accounts for decoder Jacobian)
        #   - Proposal asymmetry: log(q_rev/q_fwd) (MALA correction)
        log_energy_term = -(prop_energy - current_energy)
        log_vol_term = prop_log_vol - current_log_vol
        log_correction = log_q_rev - log_q_fwd
        
        log_alpha = log_energy_term + log_vol_term + log_correction
        
        # =====================================================================
        # 4. Metropolis-Hastings accept/reject
        # =====================================================================
        accept_prob = torch.exp(torch.clamp(log_alpha, max=0.0))
        uniform_rand = torch.rand_like(accept_prob)
        accept_mask = uniform_rand < accept_prob
        
        # =====================================================================
        # 5. Update state
        # =====================================================================
        # Apply mask to accept/reject each sample independently
        mask_expanded = accept_mask.view(-1, 1).expand_as(x)
        
        x = torch.where(mask_expanded, x_prop, x)
        score = torch.where(mask_expanded, score_prop, score)
        current_energy = torch.where(accept_mask, prop_energy, current_energy)
        current_log_vol = torch.where(accept_mask, prop_log_vol, current_log_vol)

        # Detach to prevent computation graph from growing
        x = x.detach().requires_grad_(True)
        
        # =====================================================================
        # 6. Store history and check stopping condition
        # =====================================================================
        if return_history:
            history.append(x.detach().clone())
            
        if stop_func is not None and stop_func(x):
            print(f'Early stopping at step {step}')
            break

    # Return results
    if return_history:
        return torch.stack(history, dim=0).detach()
        
    return x.detach()

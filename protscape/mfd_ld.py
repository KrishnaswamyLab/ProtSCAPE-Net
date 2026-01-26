import torch
from tqdm import tqdm

def mfd_ula(f_fn, denoiser, x_init, num_steps, step_size, step_size_noise=None, step_size_rescale=None, return_history=False, stop_func=None, seed=None):
    """
    Perform Langevin dynamics sampling using the negative log-probability function f(x).
    Unadjusted Langevin algorithm extended to batched inputs.

    Args:
        f_fn (callable): Function that computes f(x) = -log p(x) at x for batched inputs.
        x_init (torch.Tensor): Initial sample tensor of shape [batch_size, dim].
        num_steps (int): Number of Langevin steps to perform.
        step_size (float): Step size for the Langevin updates.

    Returns:
        torch.Tensor: The final samples after applying Langevin dynamics.
    """
    if seed is not None:
        torch.manual_seed(seed)
    x = x_init.clone().detach().requires_grad_(True)
    if return_history:
        history = []
    for step in tqdm(range(num_steps)):
        if f_fn is not None:
            # Compute f(x) for each sample in the batch
            f_x = f_fn(x)  # f_x should be of shape [batch_size]
            
            # Compute the gradient of f(x) with respect to x
            grad_outputs = torch.ones_like(f_x)
            grad_f = torch.autograd.grad(f_x, x, grad_outputs=grad_outputs, create_graph=True)[0]
            
            # The score is the negative gradient of f(x)
            score = -grad_f

        # Generate Gaussian noise
        noise = torch.randn_like(x)
        
        # Compute sqrt of step size
        if step_size_noise is None:
            step_size_noise = step_size
        sqrt_step_size = (2 * step_size_noise) ** 0.5
        
        # Update the sample using Langevin dynamics
        if f_fn is not None:
            delta_x = step_size * score + sqrt_step_size * noise
        else:
            delta_x = sqrt_step_size * noise
        if step_size_rescale is not None:
            assert step_size_rescale > 0.
            delta_x = step_size_rescale * torch.nn.functional.normalize(delta_x, p=2, dim=1)
        x = x + delta_x
        x = denoiser(x)
        # Detach to prevent computation graph from growing and re-enable gradients
        x = x.detach().requires_grad_(True)
        if return_history:
            history.append(x)
        if stop_func is not None and stop_func(x):
            print('Early stopping at step', step)
            break
    if return_history:
        return torch.stack(history, dim=0).detach()
        
    return x.detach()

def mfd_ula_force_momentum(f_fn, denoiser, x_init, num_steps, step_size, step_size_noise=None, step_size_rescale=1e-4, return_history=False, stop_func=None, seed=None, momentum=0.0):
    """
    Perform Langevin dynamics sampling using the negative log-probability function f(x).
    Unadjusted Langevin algorithm extended to batched inputs.

    Args:
        f_fn (callable): Function that computes grad f(x), where f(x) = -log p(x) at x for batched inputs.
        x_init (torch.Tensor): Initial sample tensor of shape [batch_size, dim].
        num_steps (int): Number of Langevin steps to perform.
        step_size (float): Step size for the Langevin updates.

    Returns:
        torch.Tensor: The final samples after applying Langevin dynamics.
    """
    if momentum != 0.0:
        assert 0 < momentum < 1, "momentum must be between 0 and 1"
    
    if seed is not None:
        torch.manual_seed(seed)
    x = x_init.clone().detach().requires_grad_(True)
    if return_history:
        history = []
    
    prev_grad_f = None
    for step in tqdm(range(num_steps)):
        # Compute f(x) for each sample in the batch
        grad_f = f_fn(x)
        
        # Apply momentum if enabled
        if momentum != 0.0 and prev_grad_f is not None:
            grad_f = momentum * prev_grad_f + (1 - momentum) * grad_f
        
        # Store current gradient for next iteration
        if momentum != 0.0:
            prev_grad_f = grad_f.detach().clone()
        
        # The score is the negative gradient of f(x)
        score = -grad_f

        # Generate Gaussian noise
        noise = torch.randn_like(x)
        if step_size_noise is None:
            step_size_noise = step_size
        # Compute sqrt of step size
        sqrt_step_size = (2 * step_size_noise) ** 0.5

        delta_x = step_size * score + sqrt_step_size * noise
        if step_size_rescale is not None:
            assert step_size_rescale > 0.
            delta_x = step_size_rescale * torch.nn.functional.normalize(delta_x, p=2, dim=1)
        x = x + delta_x

        x = denoiser(x)
        # Detach to prevent computation graph from growing and re-enable gradients
        x = x.detach().requires_grad_(True)
        if return_history:
            history.append(x)
        if stop_func is not None and stop_func(x):
            print('Early stopping at step', step)
            break
    if return_history:
        return torch.stack(history, dim=0).detach()
        
    return x.detach()


"""
Below are DEPRECATED!
"""

def mfd_ula_force(f_fn, denoiser, x_init, num_steps, step_size, step_size_noise=None, step_size_rescale=1e-4, return_history=False, stop_func=None, seed=None):
    """
    Perform Langevin dynamics sampling using the negative log-probability function f(x).
    Unadjusted Langevin algorithm extended to batched inputs.

    Args:
        f_fn (callable): Function that computes grad f(x), where f(x) = -log p(x) at x for batched inputs.
        x_init (torch.Tensor): Initial sample tensor of shape [batch_size, dim].
        num_steps (int): Number of Langevin steps to perform.
        step_size (float): Step size for the Langevin updates.

    Returns:
        torch.Tensor: The final samples after applying Langevin dynamics.
    """
    if seed is not None:
        torch.manual_seed(seed)
    x = x_init.clone().detach().requires_grad_(True)
    if return_history:
        history = []
    for step in tqdm(range(num_steps)):
        # Compute f(x) for each sample in the batch
        grad_f = f_fn(x)
        
        # The score is the negative gradient of f(x)
        score = -grad_f

        # Generate Gaussian noise
        noise = torch.randn_like(x)
        if step_size_noise is None:
            step_size_noise = step_size
        # Compute sqrt of step size
        sqrt_step_size = (2 * step_size_noise) ** 0.5

        delta_x = step_size * score + sqrt_step_size * noise
        if step_size_rescale is not None:
            assert step_size_rescale > 0.
            delta_x = step_size_rescale * torch.nn.functional.normalize(delta_x, p=2, dim=1)
        x = x + delta_x

        x = denoiser(x)
        # Detach to prevent computation graph from growing and re-enable gradients
        x = x.detach().requires_grad_(True)
        if return_history:
            history.append(x)
        if stop_func is not None and stop_func(x):
            print('Early stopping at step', step)
            break
    if return_history:
        return torch.stack(history, dim=0).detach()
        
    return x.detach()

def mfd_gradient_ascent(f_fn, denoiser, x_init, num_steps, step_size, return_history=False, stop_func=None, seed=None):
    """
    Perform Langevin dynamics sampling using the negative log-probability function f(x).
    Unadjusted Langevin algorithm extended to batched inputs.

    Args:
        f_fn (callable): Function that computes f(x) = -log p(x) at x for batched inputs.
        x_init (torch.Tensor): Initial sample tensor of shape [batch_size, dim].
        num_steps (int): Number of Langevin steps to perform.
        step_size (float): Step size for the Langevin updates.

    Returns:
        torch.Tensor: The final samples after applying Langevin dynamics.
    """
    if seed is not None:
        torch.manual_seed(seed)
    x = x_init.clone().detach().requires_grad_(True)
    if return_history:
        history = []
    for step in tqdm(range(num_steps)):
        # Compute f(x) for each sample in the batch
        f_x = f_fn(x)  # f_x should be of shape [batch_size]
        
        # Compute the gradient of f(x) with respect to x
        grad_outputs = torch.ones_like(f_x)
        grad_f = torch.autograd.grad(f_x, x, grad_outputs=grad_outputs, create_graph=True)[0]
        
        # The score is the negative gradient of f(x)
        score = -grad_f

        # Generate Gaussian noise
        # noise = torch.randn_like(x)
        
        # Compute sqrt of step size
        # sqrt_step_size = (2 * step_size) ** 0.5
        
        # Update the sample using Langevin dynamics
        x = x + step_size * score #+ sqrt_step_size * noise
        x = denoiser(x)
        # Detach to prevent computation graph from growing and re-enable gradients
        x = x.detach().requires_grad_(True)
        if return_history:
            history.append(x)
        if stop_func is not None and stop_func(x):
            print('Early stopping at step', step)
            break
    if return_history:
        return torch.stack(history, dim=0).detach()
        
    return x.detach()

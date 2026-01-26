"""
Decoding and action computation functions for generation trajectories.
"""

from typing import Optional, Tuple, Callable, Any, List
import numpy as np
import torch
import MDAnalysis as mda
from utils.geometry import lift_sel_to_full_nm


NM_TO_ANG = 10.0


@torch.no_grad()
def decode_paths_to_xyz_nm(
    model: torch.nn.Module,
    traj: torch.Tensor,         # (T,B,D)
    device: torch.device,
    decode_every: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode latent trajectories to xyz coordinates.
    
    Args:
        model: Decoder model with reconstruct_xyz method
        traj: Latent space trajectories, shape (T, B, D)
        device: torch device to use
        decode_every: Decode every Nth timestep (for subsampling)
        
    Returns:
        Tuple of (xyz_nm, t_idx) where:
        - xyz_nm: Decoded coordinates in nm, shape (Tsub, B, N, 3)
        - t_idx: Timestep indices that were decoded
    """
    model = model.to(device).eval()
    if traj.ndim != 3:
        raise ValueError(f"Expected traj (T,B,D), got {tuple(traj.shape)}")

    T, B, D = traj.shape
    t_idx = np.arange(0, T, max(1, decode_every))
    traj_sub = traj[t_idx]  # (Tsub,B,D)

    z_flat = traj_sub.reshape(traj_sub.shape[0] * B, D).to(device)  # (Tsub*B,D)

    decoded = []
    bs = 64
    for i in range(0, z_flat.shape[0], bs):
        z_b = z_flat[i:i+bs]
        x = model.reconstruct_xyz(z_b)
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3 and x_np.shape[-1] == 3:
            pos = x_np
        else:
            pos = x_np.reshape(x_np.shape[0], -1, 3)
        decoded.append(pos)

    xyz = np.concatenate(decoded, axis=0)  # (Tsub*B,N,3)
    xyz = xyz.reshape(traj_sub.shape[0], B, xyz.shape[1], 3)  # (Tsub,B,N,3)
    return xyz.astype(np.float64), t_idx.astype(np.int64)


def compute_om_action_and_prob(
    xyz_paths_nm: np.ndarray,     # (T,B,N,3) selection or full
    force_eval: Any,              # OpenMMForceEvaluator
    pdb_path: str,
    use_full_system: bool,
    sel_idx: Optional[np.ndarray],
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Onsager-Machlup action and relative path probability in decoded space.
    
    OM discretization (overdamped Langevin):
      S ≈ (dt/4) Σ || (x_{t+1}-x_t)/dt + ∇U(x_t) ||^2
    with ∇U = -F (OpenMM returns F = -∇U).

    Key point:
      OpenMM ForceEvaluator was built on a specific system size (expected_n_atoms).
      If xyz_paths_nm is a selection (Nsel), we MUST lift to full coords before calling OpenMM,
      then optionally restrict forces back to selection for the OM norm.

    Behavior controlled by `use_full_system`:
      - If True: compute OM action on FULL system (dx and gradU both full).
      - If False: compute OM action on SELECTION only, but forces are still computed on full system
        (lift -> OpenMM -> restrict) because OpenMM requires full positions.

    Args:
        xyz_paths_nm: Trajectory coordinates, shape (T, B, N, 3) in nanometers
        force_eval: OpenMMForceEvaluator instance
        pdb_path: Path to reference PDB
        use_full_system: Whether to compute action on full system vs selection
        sel_idx: Selection atom indices (required if xyz_paths_nm is selection coords)
        dt: Timestep size
        
    Returns:
        Tuple of (actions, rel_prob) where:
        - actions: OM action for each path, shape (B,)
        - rel_prob: Relative probability (normalized), shape (B,)
    """
    if xyz_paths_nm.ndim != 4:
        raise ValueError(f"xyz_paths_nm must be (T,B,N,3); got {xyz_paths_nm.shape}")

    u_ref = mda.Universe(pdb_path)
    full_ref_nm = (u_ref.atoms.positions.astype(np.float64) / NM_TO_ANG)

    T, B, N, C = xyz_paths_nm.shape
    if C != 3:
        raise ValueError(f"xyz_paths_nm last dim must be 3; got {C}")

    expected_n = getattr(force_eval, "expected_n_atoms", None)  # may be None
    if expected_n is not None and full_ref_nm.shape[0] != expected_n:
        raise RuntimeError(
            f"PDB has {full_ref_nm.shape[0]} atoms but OpenMM expects {expected_n}. "
            "Your OpenMM system was likely built from a different PDB."
        )

    is_full_input = (expected_n is not None and N == expected_n)

    if not is_full_input:
        # selection input (e.g., N=220). Need sel_idx to lift into full system.
        if sel_idx is None:
            raise RuntimeError(
                "xyz_paths_nm appears to be a selection (not full-system), but sel_idx is None. "
                "Provide sel_idx (sel_atom_indices) so we can lift to full coords for OpenMM."
            )
        sel_idx = np.asarray(sel_idx, dtype=int)
        if sel_idx.ndim != 1:
            raise ValueError(f"sel_idx must be 1D; got shape {sel_idx.shape}")
        if sel_idx.shape[0] != N:
            raise RuntimeError(
                f"Selection mismatch: xyz_paths_nm has N={N} atoms but sel_idx has {sel_idx.shape[0]} indices."
            )

    actions = np.zeros((B,), dtype=np.float64)

    for b in range(B):
        S = 0.0
        for t in range(T - 1):
            x_t = xyz_paths_nm[t, b].astype(np.float64, copy=False)      # (N,3)
            x_tp1 = xyz_paths_nm[t + 1, b].astype(np.float64, copy=False)

            if is_full_input:
                # We already have full-system coords; can call OpenMM directly.
                F_full = force_eval.compute_forces(x_t)  # (Nfull,3)

                if use_full_system:
                    # action on full system
                    gradU = (-F_full).reshape(-1)
                    dx = (x_tp1 - x_t).reshape(-1)
                else:
                    # action only on selection components (but force computed from full)
                    if sel_idx is None:
                        raise RuntimeError(
                            "use_full_system=False requires sel_idx when xyz_paths_nm is full-system, "
                            "so we know which subset to score."
                        )
                    F_sel = F_full[sel_idx]
                    gradU = (-F_sel).reshape(-1)
                    dx = (x_tp1[sel_idx] - x_t[sel_idx]).reshape(-1)

            else:
                # Input is selection coords: must lift to full for OpenMM.
                x_full = lift_sel_to_full_nm(x_t, full_ref_nm, sel_idx)   # (Nfull,3)
                F_full = force_eval.compute_forces(x_full)                # (Nfull,3)

                if use_full_system:
                    # action on full system: also lift x_tp1 so dx is full
                    x_full_tp1 = lift_sel_to_full_nm(x_tp1, full_ref_nm, sel_idx)
                    gradU = (-F_full).reshape(-1)
                    dx = (x_full_tp1 - x_full).reshape(-1)
                else:
                    # action on selection only
                    F_sel = F_full[sel_idx]
                    gradU = (-F_sel).reshape(-1)
                    dx = (x_tp1 - x_t).reshape(-1)

            xdot = dx / float(dt)
            v = xdot + gradU
            S += (float(dt) / 4.0) * float(np.dot(v, v))

        actions[b] = S

    a0 = float(np.nanmin(actions))
    w = np.exp(-(actions - a0))
    rel_prob = w / (w.sum() + 1e-12)
    return actions, rel_prob

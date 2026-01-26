"""
Utility functions for geometric computations, including Kabsch alignment and RMSD calculations.
"""

import numpy as np
from typing import Optional, Tuple

def kabsch_align_np(P: np.ndarray, Q: np.ndarray):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: P {P.shape}, Q {Q.shape}")

    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt

    P_aligned = Pc @ R + Q.mean(axis=0, keepdims=True)
    return P_aligned, R


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    P_aligned, _ = kabsch_align_np(P, Q)
    diff = P_aligned - Q
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


def mse_xyz(P: np.ndarray, Q: np.ndarray) -> float:
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    return float(np.mean((P - Q) ** 2))


def lift_sel_to_full_nm(
    x_nm: np.ndarray,
    full_ref_nm: np.ndarray,
    sel_idx: Optional[np.ndarray],
) -> np.ndarray:
    """
    Lift selection coordinates to full system coordinates using reference structure.
    
    Args:
        x_nm: Selection or full coordinates, shape (N_sel_or_full, 3) in nm
        full_ref_nm: Reference full system coordinates, shape (N_full, 3) in nm
        sel_idx: Selection atom indices, or None if x_nm is already full system
        
    Returns:
        Full system coordinates with selected atoms replaced by x_nm
    """
    if sel_idx is None:
        return x_nm
    if x_nm.shape[0] == full_ref_nm.shape[0]:
        return x_nm
    if x_nm.shape[0] != sel_idx.shape[0]:
        raise RuntimeError(f"Cannot lift: x has {x_nm.shape[0]} atoms but sel has {sel_idx.shape[0]}")
    out = full_ref_nm.copy()
    out[sel_idx] = x_nm
    return out
"""
OpenMM evaluators for computing forces and energies on molecular structures.
"""

from typing import Any, Callable, List, Optional
import numpy as np
import torch
from torch.autograd.functional import jacobian
import MDAnalysis as mda
from openmm.unit import nanometer, kilojoule_per_mole


NM_TO_ANG = 10.0


class OpenMMForceEvaluator:
    """Evaluates forces using OpenMM for a given structure."""
    
    def __init__(self, setup_simulation_fn: Callable, pdb_path: str, first_data: Any):
        """
        Initialize force evaluator.
        
        Args:
            setup_simulation_fn: Function that returns OpenMM simulation object
            pdb_path: Path to reference PDB file
            first_data: First graph object from dataset
        """
        sim_out = setup_simulation_fn(pdb_path)
        self.simulation = sim_out[0] if isinstance(sim_out, tuple) else sim_out
        self.first_data = first_data

        expected_n = None
        try:
            expected_n = self.simulation.topology.getNumAtoms()
        except Exception:
            try:
                expected_n = self.simulation.context.getSystem().getNumParticles()
            except Exception:
                expected_n = None
        self.expected_n_atoms = expected_n

    def compute_forces(self, positions_nm: np.ndarray) -> np.ndarray:
        """
        Compute forces at given positions.
        
        Args:
            positions_nm: Atomic positions in nanometers, shape (N, 3)
            
        Returns:
            Forces at those positions, shape (N, 3)
        """
        from openmm import unit
        if self.expected_n_atoms is not None and positions_nm.shape[0] != self.expected_n_atoms:
            raise RuntimeError(
                f"OpenMM expected {self.expected_n_atoms} atoms but got {positions_nm.shape[0]}."
            )
        qpos = unit.Quantity(positions_nm, unit=nanometer)
        self.simulation.context.setPositions(qpos)
        state = self.simulation.context.getState(getForces=True)
        forces = state.getForces(asNumpy=True)
        return np.asarray(forces)


class OpenMMEnergyEvaluator:
    """
    Computes potential energy (kJ/mol) for a decoded structure, handling:
      - full-system decoded coords (n_full,3)
      - selection-only decoded coords (n_sel,3) via lifting using sel_atom_indices into reference full coords
    """
    
    def __init__(self, setup_simulation_fn: Callable, pdb_path: str, first_data: Any):
        """
        Initialize energy evaluator.
        
        Args:
            setup_simulation_fn: Function that returns OpenMM simulation object
            pdb_path: Path to reference PDB file
            first_data: First graph object from dataset
        """
        sim_out = setup_simulation_fn(pdb_path)
        self.simulation = sim_out[0] if isinstance(sim_out, tuple) else sim_out
        self.first_data = first_data

        if hasattr(first_data, "sel_atom_indices"):
            self.sel_idx = first_data.sel_atom_indices.detach().cpu().numpy().astype(int)
        else:
            self.sel_idx = None

        u_ref = mda.Universe(pdb_path)
        self.full_ref_A = u_ref.atoms.positions.astype(np.float64)
        self.full_ref_nm = self.full_ref_A / NM_TO_ANG

        expected_n = None
        try:
            expected_n = self.simulation.topology.getNumAtoms()
        except Exception:
            try:
                expected_n = self.simulation.context.getSystem().getNumParticles()
            except Exception:
                expected_n = None
        self.expected_n_atoms = expected_n

        if self.expected_n_atoms is not None and self.expected_n_atoms != self.full_ref_nm.shape[0]:
            raise RuntimeError(
                f"PDB has {self.full_ref_nm.shape[0]} atoms but OpenMM expects {self.expected_n_atoms}. "
                "You likely built the OpenMM system from a different PDB than you're passing here."
            )

    def _ensure_nanometers(self, positions: np.ndarray) -> np.ndarray:
        """Convert positions from Angstroms to nanometers if needed."""
        if np.max(np.abs(positions)) > 5.0:
            return positions * 0.1
        return positions

    def compute_energy_kjmol(self, decoded_positions_nm_or_A: np.ndarray) -> float:
        """
        Compute potential energy at given positions.
        
        Args:
            decoded_positions_nm_or_A: Coordinates in nanometers or angstroms, shape (N_sel_or_full, 3)
            
        Returns:
            Potential energy in kJ/mol
        """
        pos_nm = self._ensure_nanometers(decoded_positions_nm_or_A.astype(np.float64))

        if self.expected_n_atoms is not None and pos_nm.shape[0] == self.expected_n_atoms:
            full_pos_nm = pos_nm
        elif self.sel_idx is not None and pos_nm.shape[0] == self.sel_idx.shape[0]:
            full_pos_nm = self.full_ref_nm.copy()
            full_pos_nm[self.sel_idx] = pos_nm
        else:
            raise RuntimeError(
                f"Energy eval: decoded coords have {pos_nm.shape[0]} atoms; "
                f"selection has {None if self.sel_idx is None else self.sel_idx.shape[0]} atoms; "
                f"OpenMM expects {self.expected_n_atoms} atoms."
            )

        from openmm import unit
        qpos = unit.Quantity(full_pos_nm, unit=nanometer)
        self.simulation.context.setPositions(qpos)
        state = self.simulation.context.getState(getEnergy=True)
        pe = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        return float(pe)


def build_grad_fn_openmm(
    model: torch.nn.Module,
    dataset: List[Any],
    setup_simulation_fn: Callable,
    pdb_path: str,
    device: torch.device,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a function that computes the gradient of potential energy in latent space.
    
    Uses OpenMM forces and Jacobian to compute grad_z of potential energy.
    
    Args:
        model: Decoder model with reconstruct_xyz method
        dataset: List of graph objects from dataset
        setup_simulation_fn: Function that returns OpenMM simulation
        pdb_path: Path to reference PDB
        device: torch device to use
        
    Returns:
        grad_fn(z_batch) function that returns gradients, shape (B, D)
    """
    model = model.to(device).eval()
    data0 = dataset[0]
    force_eval = OpenMMForceEvaluator(setup_simulation_fn, pdb_path=pdb_path, first_data=data0)

    if hasattr(data0, "sel_atom_indices"):
        sel_idx = data0.sel_atom_indices.detach().cpu().numpy().astype(int)
    else:
        sel_idx = None

    u_ref = mda.Universe(pdb_path)
    full_pos_ref_A = u_ref.atoms.positions.astype(np.float64)
    full_pos_ref_nm = full_pos_ref_A / NM_TO_ANG

    if force_eval.expected_n_atoms is not None and force_eval.expected_n_atoms != full_pos_ref_nm.shape[0]:
        raise RuntimeError(
            f"PDB has {full_pos_ref_nm.shape[0]} atoms but OpenMM expects {force_eval.expected_n_atoms}. "
            "You likely built the OpenMM system from a different PDB than you're passing here."
        )

    def ensure_nanometers(positions: np.ndarray) -> np.ndarray:
        if np.max(np.abs(positions)) > 5.0:
            return positions * 0.1
        return positions

    def grad_fn(z_batch: torch.Tensor) -> torch.Tensor:
        if not isinstance(z_batch, torch.Tensor):
            z_batch = torch.tensor(z_batch, dtype=torch.float)
        z_batch = z_batch.detach().to(device)

        grads_out: List[torch.Tensor] = []
        for zi in z_batch:
            zi = zi.unsqueeze(0).requires_grad_(True)

            x = model.reconstruct_xyz(zi)
            x_np = x.detach().cpu().numpy()
            if x_np.ndim == 3 and x_np.shape[-1] == 3:
                pos = x_np[0]
            else:
                pos = x_np.reshape(-1, 3)

            pos_nm = ensure_nanometers(pos)

            if force_eval.expected_n_atoms is not None and pos_nm.shape[0] == force_eval.expected_n_atoms:
                forces_used = force_eval.compute_forces(pos_nm)
            elif sel_idx is not None and pos_nm.shape[0] == sel_idx.shape[0]:
                full_pos_nm = full_pos_ref_nm.copy()
                full_pos_nm[sel_idx] = pos_nm
                forces_full = force_eval.compute_forces(full_pos_nm)
                forces_used = forces_full[sel_idx]
            else:
                raise RuntimeError(
                    f"Decoded coords have {pos_nm.shape[0]} atoms; "
                    f"selection has {None if sel_idx is None else sel_idx.shape[0]} atoms; "
                    f"OpenMM expects {force_eval.expected_n_atoms} atoms. "
                    "Cannot determine mapping."
                )

            grad_x = torch.as_tensor(forces_used.reshape(-1), dtype=torch.float, device=device)
            def decode_fn(latent: torch.Tensor) -> torch.Tensor:
                return model.reconstruct_xyz(latent)

            J = jacobian(decode_fn, zi, create_graph=False)
            J = torch.as_tensor(J).detach()
            J_view = J.reshape(-1, J.shape[-1]) if J.ndim > 2 else J

            if grad_x.numel() != J_view.shape[0]:
                try:
                    J_view = J.reshape(grad_x.numel(), -1)
                except Exception as e:
                    raise RuntimeError(
                        f"Jacobian/force mismatch: grad_x={grad_x.numel()} vs J={tuple(J.shape)}. {e}"
                    )

            # OpenMM gives F = -∇U ; so ∇U = -F. We want grad_z = ∇U^T (dx/dz)
            grad_z = ((-grad_x).unsqueeze(0) @ J_view.to(device)).squeeze(0)
            grads_out.append(grad_z.detach().cpu())

        return torch.stack(grads_out, dim=0)

    return grad_fn

def minimize_structure_openmm(
    simulation: Any,
    positions_nm: np.ndarray,
    max_iterations: int = 100,
) -> np.ndarray:
    """
    Minimize a structure using OpenMM's LocalEnergyMinimizer.
    
    Args:
        simulation: OpenMM simulation object
        positions_nm: Initial positions in nanometers, shape (N, 3)
        max_iterations: Maximum number of minimization iterations
        
    Returns:
        Minimized positions in nanometers, shape (N, 3)
    """
    from openmm import unit, LocalEnergyMinimizer
    
    qpos = unit.Quantity(positions_nm, unit=nanometer)
    simulation.context.setPositions(qpos)
    
    # Minimize
    LocalEnergyMinimizer.minimize(simulation.context, maxIterations=max_iterations)
    
    # Get minimized positions
    state = simulation.context.getState(getPositions=True)
    min_positions = state.getPositions(asNumpy=True)
    return np.asarray(min_positions)


def compute_energy_change(
    energy_eval: OpenMMEnergyEvaluator,
    positions_nm: np.ndarray,
    max_iterations: int = 100,
) -> tuple[float, float, float]:
    """
    Compute energy change after minimization (ΔE).
    
    Args:
        energy_eval: OpenMMEnergyEvaluator instance
        positions_nm: Initial positions in nanometers, shape (N, 3)
        max_iterations: Maximum minimization iterations
        
    Returns:
        Tuple of (initial_energy, final_energy, delta_E) in kJ/mol
    """
    # Initial energy
    E_initial = energy_eval.compute_energy_kjmol(positions_nm)
    
    # Minimize
    pos_min_nm = minimize_structure_openmm(
        energy_eval.simulation,
        positions_nm,
        max_iterations=max_iterations,
    )
    
    # Final energy
    E_final = energy_eval.compute_energy_kjmol(pos_min_nm)
    
    delta_E = E_final - E_initial
    
    return E_initial, E_final, delta_E


def compute_interframe_rmsd(xyz_path_nm: np.ndarray) -> np.ndarray:
    """
    Compute RMSD between successive frames along a path.
    
    Args:
        xyz_path_nm: Path coordinates in nanometers, shape (T, N, 3)
        
    Returns:
        Array of RMSDs between consecutive frames in Angstroms, shape (T-1,)
    """
    from utils.geometry import kabsch_rmsd
    
    if xyz_path_nm.shape[0] < 2:
        return np.array([])
    
    rmsds = []
    xyz_path_A = xyz_path_nm * NM_TO_ANG
    
    for i in range(xyz_path_A.shape[0] - 1):
        rmsd = kabsch_rmsd(xyz_path_A[i], xyz_path_A[i + 1])
        rmsds.append(rmsd)
    
    return np.array(rmsds)
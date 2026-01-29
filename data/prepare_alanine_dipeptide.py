"""
prepare_alanine_dipeptide.py

Prepare Alanine Dipeptide MD trajectory as PyTorch Geometric graphs,
following the same format as ATLAS dataset processing.

Graph structure:
  - nodes = atoms (all atoms in alanine dipeptide)
  - edges = bonds (undirected)
  - node features x = [atomic_number, residue_index, amino_acid_index, x, y, z]  (n, 6)
  - positions pos = xyz in nm (n, 3)
  - edge_attr = [eq_len_nm, cur_len_nm, delta_len_nm, bond_type_onehot(4)]  (E, 7)
  - energy = potential energy (kJ/mol) computed via OpenMM
  - y = dihedral angles [phi, psi] for conformational state tracking
  - time = trajectory timepoint

Alanine dipeptide has two major conformational states:
  - C7eq (extended): local minimum at Phi ~ -80°, Psi ~ 80°
  - C7ax (compact): local minimum at Phi ~ 80°, Psi ~ -80°

Usage:
  python prepare_alanine_dipeptide.py [--input_dir <DIR>] [--output <FILE>] [--selection <SEL>]

Example:
  python prepare_alanine_dipeptide.py
  python prepare_alanine_dipeptide.py --selection "all"
  python prepare_alanine_dipeptide.py --output alanine_dipeptide_graphs.pkl
"""

import mdtraj as md
import numpy as np
import torch
import torch_geometric.data as Data
import pickle
from tqdm import tqdm
import argparse
import os
from pathlib import Path

from openmm.app import *
from openmm import *
from openmm.unit import *


# ----------------------------
# OpenMM setup for alanine dipeptide
# ----------------------------
def setup_simulation(pdb_path: str):
    """
    Setup OpenMM simulation for alanine dipeptide.
    Using implicit solvent (vacuum or implicit) since the dataset is in vacuum.
    """
    pdb = PDBFile(pdb_path)
    
    # Use implicit solvent forcefield for vacuum simulation
    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=NoCutoff,  # No cutoff for vacuum
        constraints=HBonds,
        rigidWater=True
    )
    
    integrator = LangevinMiddleIntegrator(
        300 * kelvin,
        1.0 / picosecond,
        0.002 * picoseconds
    )
    
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    
    return simulation, system


def load_trajectory(dcd_path: str, pdb_path: str):
    """Load DCD trajectory with PDB topology."""
    return md.load(dcd_path, top=pdb_path)


# ----------------------------
# Amino acid vocabulary (same as ATLAS)
# ----------------------------
AA20 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
]
AA_TO_IDX = {a: i for i, a in enumerate(AA20)}
AA_UNK = len(AA20)  # 20


# ----------------------------
# Node features (same as ATLAS)
# ----------------------------
def atomic_number_feature(traj: md.Trajectory) -> torch.Tensor:
    """(n,1) atomic numbers Z. Unknown -> 0."""
    Z = []
    for atom in traj.topology.atoms:
        if atom.element is None:
            Z.append(0)
        else:
            Z.append(atom.element.atomic_number)
    return torch.tensor(np.asarray(Z, dtype=np.float32)[:, None], dtype=torch.float32)


def residue_number_feature(traj: md.Trajectory) -> torch.Tensor:
    """(n,1) raw residue indices 0..n_res-1."""
    res_idx = np.array([atom.residue.index for atom in traj.topology.atoms], dtype=np.float32)
    return torch.tensor(res_idx[:, None], dtype=torch.float32)


def amino_acid_index_feature(traj: md.Trajectory) -> torch.Tensor:
    """(n,1) aa index in [0..20] (UNK=20)."""
    aa_idx = []
    for atom in traj.topology.atoms:
        res = atom.residue.name.upper()
        aa_idx.append(AA_TO_IDX.get(res, AA_UNK))
    return torch.tensor(np.asarray(aa_idx, dtype=np.float32)[:, None], dtype=torch.float32)


# ----------------------------
# Bond graph construction (same as ATLAS)
# ----------------------------
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
BOND_TO_IDX = {b: i for i, b in enumerate(BOND_TYPES)}


def _infer_bond_type_mdtraj(bond) -> str:
    """Infer bond type from MDTraj bond object."""
    # Aromatic flag
    if hasattr(bond, "is_aromatic") and bool(getattr(bond, "is_aromatic")):
        return "AROMATIC"
    
    # Numeric order
    if hasattr(bond, "order"):
        try:
            o = int(getattr(bond, "order"))
            if o == 1: return "SINGLE"
            if o == 2: return "DOUBLE"
            if o == 3: return "TRIPLE"
        except Exception:
            pass
    
    # String type
    for attr in ["type", "bond_type"]:
        if hasattr(bond, attr):
            v = getattr(bond, attr)
            if v is None:
                continue
            s = str(v).upper()
            if "AROM" in s:
                return "AROMATIC"
            if "SING" in s:
                return "SINGLE"
            if "DOUB" in s:
                return "DOUBLE"
            if "TRIP" in s:
                return "TRIPLE"
    
    return None


def build_bond_graph_for_subset(topology, atom_indices, make_undirected=True):
    """
    Build edge_index and edge attributes for selected atoms.
    
    Returns:
      edge_index: (2, E)
      edge_pairs: (E, 2) int64, node indices i,j
      bond_type_oh: (E, 4) float32 onehot
    """
    atom_indices = np.asarray(atom_indices, dtype=int)
    sel_set = set(atom_indices.tolist())
    old_to_new = {old: new for new, old in enumerate(atom_indices.tolist())}
    
    edges = []
    pairs = []
    type_oh = []
    
    for bond in topology.bonds:
        a1, a2 = bond
        i_old, j_old = int(a1.index), int(a2.index)
        if (i_old in sel_set) and (j_old in sel_set):
            i = old_to_new[i_old]
            j = old_to_new[j_old]
            
            btype = _infer_bond_type_mdtraj(bond)
            oh = np.zeros((4,), dtype=np.float32)
            if btype is not None and btype in BOND_TO_IDX:
                oh[BOND_TO_IDX[btype]] = 1.0
            
            # Directed edge i->j
            edges.append((i, j))
            pairs.append((i, j))
            type_oh.append(oh)
            
            if make_undirected:
                edges.append((j, i))
                pairs.append((j, i))
                type_oh.append(oh)
    
    if len(edges) == 0:
        raise RuntimeError("No bonds found. Check topology.")
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_pairs = torch.tensor(np.asarray(pairs), dtype=torch.long)
    bond_type_oh = torch.tensor(np.asarray(type_oh), dtype=torch.float32)
    
    return edge_index, edge_pairs, bond_type_oh


def compute_equilibrium_bond_lengths(traj: md.Trajectory, edge_pairs: torch.Tensor) -> torch.Tensor:
    """Use first frame as equilibrium length (nm)."""
    xyz0 = traj.xyz[0].astype(np.float32)  # (n,3) nm
    ij = edge_pairs.cpu().numpy()
    vec = xyz0[ij[:, 0]] - xyz0[ij[:, 1]]
    d = np.sqrt((vec * vec).sum(axis=1, keepdims=True)).astype(np.float32)
    return torch.tensor(d, dtype=torch.float32)


def compute_current_bond_lengths(frame_xyz_nm: np.ndarray, edge_pairs: torch.Tensor) -> torch.Tensor:
    """Compute current bond lengths for a frame."""
    ij = edge_pairs.cpu().numpy()
    vec = frame_xyz_nm[ij[:, 0]] - frame_xyz_nm[ij[:, 1]]
    d = np.sqrt((vec * vec).sum(axis=1, keepdims=True)).astype(np.float32)
    return torch.tensor(d, dtype=torch.float32)


# ----------------------------
# Dihedral angles for alanine dipeptide
# ----------------------------
def compute_dihedral_angles(traj: md.Trajectory, frame_idx: int) -> tuple:
    """
    Compute phi and psi dihedral angles for alanine dipeptide.
    
    Returns:
        (phi, psi) in degrees
    """
    frame = traj[frame_idx]
    
    # Standard backbone dihedral indices for alanine dipeptide
    # Phi: C(-1) - N - CA - C
    # Psi: N - CA - C - N(+1)
    
    # Compute all phi/psi angles (mdtraj returns in radians)
    phi_indices, phi_angles = md.compute_phi(frame)
    psi_indices, psi_angles = md.compute_psi(frame)
    
    # Convert to degrees
    phi = np.rad2deg(phi_angles[0, 0]) if phi_angles.size > 0 else 0.0
    psi = np.rad2deg(psi_angles[0, 0]) if psi_angles.size > 0 else 0.0
    
    return float(phi), float(psi)


# ----------------------------
# Graph per frame
# ----------------------------
def create_pyg_graph(
    traj: md.Trajectory,
    sel_atom_indices: np.ndarray,
    frame_idx: int,
    simulation: Simulation,
    atomic_number: torch.Tensor,
    residue_number: torch.Tensor,
    amino_acid_index: torch.Tensor,
    edge_index: torch.Tensor,
    edge_pairs: torch.Tensor,
    bond_type_oh: torch.Tensor,
    eq_len: torch.Tensor,
) -> Data.Data:
    """Create PyG graph for a single frame."""
    frame = traj[frame_idx]
    
    # Positions (nm)
    pos_np = frame.xyz[0].astype(np.float32)  # (n, 3)
    pos = torch.tensor(pos_np, dtype=torch.float32)
    
    # Node features: [atomic_number, residue_number, amino_acid_index, x, y, z]
    x = torch.cat([atomic_number, residue_number, amino_acid_index, pos], dim=1)
    
    # Edge attributes: dynamic bond lengths
    cur_len = compute_current_bond_lengths(pos_np, edge_pairs)
    delta_len = cur_len - eq_len
    edge_attr = torch.cat([eq_len, cur_len, delta_len, bond_type_oh], dim=1)
    
    # Compute potential energy using OpenMM
    simulation.context.setPositions(pos_np * nanometer)
    state = simulation.context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    
    # Compute dihedral angles (conformational state indicators)
    phi, psi = compute_dihedral_angles(traj, frame_idx)
    
    # Time (ps)
    timepoint = float(traj.time[frame_idx])
    
    graph = Data.Data(
        x=x,                        # (n, 6): [Z, res_idx, aa_idx, x, y, z]
        pos=pos,                    # (n, 3) nm
        edge_index=edge_index,      # (2, E)
        edge_attr=edge_attr,        # (E, 7)
        num_nodes=pos.shape[0],
        time=torch.tensor([timepoint], dtype=torch.float32),
        y=torch.tensor([phi, psi], dtype=torch.float32),  # [phi, psi] dihedrals
        energy=torch.tensor([potential_energy], dtype=torch.float32),
        sel_atom_indices=torch.tensor(sel_atom_indices, dtype=torch.long),
        phi=torch.tensor([phi], dtype=torch.float32),
        psi=torch.tensor([psi], dtype=torch.float32),
    )
    
    return graph


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare Alanine Dipeptide MD trajectory as PyTorch Geometric graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_alanine_dipeptide.py
  python prepare_alanine_dipeptide.py --selection "all"
  python prepare_alanine_dipeptide.py --output graphs/alanine_dipeptide_all.pkl
        """
    )
    
    parser.add_argument(
        "--input_dir",
        default="datasets/alanine_dipeptide",
        help="Input directory containing trajectory files (default: datasets/alanine_dipeptide)"
    )
    parser.add_argument(
        "--output", "-o",
        default="graphs/alanine_dipeptide_graphs.pkl",
        help="Output pickle file (default: graphs/alanine_dipeptide_graphs.pkl)"
    )
    parser.add_argument(
        "--selection",
        default="all",
        help='MDTraj atom selection (default: "all")'
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    
    # Find trajectory files
    dcd_files = list(input_path.glob("*.dcd"))
    pdb_files = list(input_path.glob("*.pdb"))
    
    if not dcd_files:
        print(f"[error] No .dcd file found in {input_path}")
        print("[info] Run download_alanine_dipeptide.py first")
        return 1
    
    if not pdb_files:
        print(f"[error] No .pdb file found in {input_path}")
        print("[info] Run download_alanine_dipeptide.py first")
        return 1
    
    dcd_path = str(dcd_files[0])
    pdb_path = str(pdb_files[0])
    
    print(f"[info] Loading trajectory: {dcd_path}")
    print(f"[info] Loading topology: {pdb_path}")
    print(f"[info] Selection: '{args.selection}'")
    
    # Setup simulation
    try:
        simulation, _ = setup_simulation(pdb_path)
        traj = load_trajectory(dcd_path, pdb_path)
    except Exception as e:
        print(f"[error] Failed to load files: {e}")
        return 1
    
    # Apply selection
    sel_atom_indices = traj.topology.select(args.selection)
    traj_sel = traj.atom_slice(sel_atom_indices)
    
    print(f"[info] Total atoms: {traj.topology.n_atoms}")
    print(f"[info] Selected atoms: {traj_sel.topology.n_atoms}")
    print(f"[info] Total frames: {traj.n_frames}")
    
    # Limit frames if specified
    n_frames = args.max_frames if args.max_frames else traj_sel.n_frames
    n_frames = min(n_frames, traj_sel.n_frames)
    print(f"[info] Processing {n_frames} frames")
    
    # Build bond graph (static)
    edge_index, edge_pairs, bond_type_oh = build_bond_graph_for_subset(
        traj_sel.topology,
        sel_atom_indices,
        make_undirected=True
    )
    print(f"[info] Edge index shape: {tuple(edge_index.shape)}")
    print(f"[info] Bond type onehot shape: {tuple(bond_type_oh.shape)}")
    
    # Static node features
    atomic_number = atomic_number_feature(traj_sel)
    residue_number = residue_number_feature(traj_sel)
    amino_acid_index = amino_acid_index_feature(traj_sel)
    
    # Equilibrium bond lengths from first frame
    eq_len = compute_equilibrium_bond_lengths(traj_sel, edge_pairs)
    
    # Create graphs for all frames
    graphs = []
    for frame_idx in tqdm(range(n_frames), desc="Building graphs"):
        g = create_pyg_graph(
            traj=traj_sel,
            sel_atom_indices=sel_atom_indices,
            frame_idx=frame_idx,
            simulation=simulation,
            atomic_number=atomic_number,
            residue_number=residue_number,
            amino_acid_index=amino_acid_index,
            edge_index=edge_index,
            edge_pairs=edge_pairs,
            bond_type_oh=bond_type_oh,
            eq_len=eq_len,
        )
        graphs.append(g)
    
    # Save graphs
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(graphs, f)
    
    print(f"\n[done] Saved {len(graphs)} graphs to {args.output}")
    print(f"[dims] x shape: {graphs[0].x.shape} (expected [n, 6])")
    print(f"[dims] pos shape: {graphs[0].pos.shape} (expected [n, 3])")
    print(f"[dims] edge_attr shape: {graphs[0].edge_attr.shape} (expected [E, 7])")
    print(f"[example] Frame 0:")
    print(f"  Energy: {graphs[0].energy.item():.6f} kJ/mol")
    print(f"  Phi: {graphs[0].phi.item():.2f}°")
    print(f"  Psi: {graphs[0].psi.item():.2f}°")
    
    # Analyze energy landscape
    energies = np.array([g.energy.item() for g in graphs])
    phis = np.array([g.phi.item() for g in graphs])
    psis = np.array([g.psi.item() for g in graphs])
    
    print(f"\n[statistics]")
    print(f"  Energy: min={energies.min():.2f}, max={energies.max():.2f}, mean={energies.mean():.2f} kJ/mol")
    print(f"  Phi: min={phis.min():.2f}°, max={phis.max():.2f}°, mean={phis.mean():.2f}°")
    print(f"  Psi: min={psis.min():.2f}°, max={psis.max():.2f}°, mean={psis.mean():.2f}°")
    
    # Identify conformational states (simple clustering)
    c7eq_mask = (phis < 0) & (psis > 0)  # Extended state
    c7ax_mask = (phis > 0) & (psis < 0)  # Compact state
    
    n_c7eq = c7eq_mask.sum()
    n_c7ax = c7ax_mask.sum()
    
    print(f"\n[conformational states]")
    print(f"  C7eq (extended, Phi<0, Psi>0): {n_c7eq} frames ({100*n_c7eq/len(graphs):.1f}%)")
    print(f"  C7ax (compact, Phi>0, Psi<0): {n_c7ax} frames ({100*n_c7ax/len(graphs):.1f}%)")
    
    if n_c7eq > 0 and n_c7ax > 0:
        print(f"  ✓ Both conformational states detected!")
        print(f"  C7eq mean energy: {energies[c7eq_mask].mean():.2f} kJ/mol")
        print(f"  C7ax mean energy: {energies[c7ax_mask].mean():.2f} kJ/mol")
    else:
        print(f"  ⚠ Warning: Only one conformational state detected in trajectory")
    
    return 0


if __name__ == "__main__":
    exit(main())

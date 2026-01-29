"""
prepare_deshaw.py

Prepare DE Shaw BPTI trajectory data as PyTorch Geometric graphs.

Graphs:
  - nodes = atoms in a fixed MDTraj selection
  - edges = bonds restricted to the selected atoms (undirected)
  - node features x = [ atomic_number | residue_index_raw | amino_acid_index ]  (n_sel, 3)
  - positions pos = xyz in nm (n_sel, 3)
  - edge_attr = [eq_len_nm, cur_len_nm, delta_len_nm, bond_type_onehot(4)]  (E, 7)
      bond_type_onehot order: [SINGLE, DOUBLE, TRIPLE, AROMATIC]
      if bond types not available -> zeros.

  - energy = OpenMM full-system potential energy (kJ/mol)
  - y = optional property (rog/sasa/none) on selected atoms
  - sel_atom_indices saved for consistent ordering

Output: pickle of list[torch_geometric.data.Data]

Usage:
  python prepare_deshaw.py [--selection <SEL>] [--property <PROP>] [--output <OUT>]

Example:
  python prepare_deshaw.py
  python prepare_deshaw.py --selection "protein and backbone" --property rog
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
# OpenMM setup (FULL system)
# ----------------------------
def setup_simulation(pdb_path: str):
    pdb = PDBFile(pdb_path)
    forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
    )

    integrator = LangevinMiddleIntegrator(300 * kelvin, 1.0 / picosecond, 0.004 * picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    return simulation, system


def load_data(xtc_path: str, pdb_path: str):
    return md.load(xtc_path, top=pdb_path)  # xyz in nm


# ----------------------------
# Amino acid vocabulary
# ----------------------------
AA20 = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
]
AA_TO_IDX = {a: i for i, a in enumerate(AA20)}
AA_UNK = len(AA20)  # 20


# ----------------------------
# Minimal node features
# ----------------------------
def atomic_number_feature(traj_sel: md.Trajectory) -> torch.Tensor:
    """(n_sel,1) atomic numbers Z. Unknown -> 0."""
    Z = []
    for atom in traj_sel.topology.atoms:
        if atom.element is None:
            Z.append(0)
        else:
            Z.append(atom.element.atomic_number)
    return torch.tensor(np.asarray(Z, dtype=np.float32)[:, None], dtype=torch.float32)


def residue_number_feature(traj_sel: md.Trajectory) -> torch.Tensor:
    """(n_sel,1) raw residue indices 0..n_res-1 (shared by atoms in residue)."""
    res_idx = np.array([atom.residue.index for atom in traj_sel.topology.atoms], dtype=np.float32)
    return torch.tensor(res_idx[:, None], dtype=torch.float32)


def amino_acid_index_feature(traj_sel: md.Trajectory) -> torch.Tensor:
    """(n_sel,1) aa index in [0..20] (UNK=20)."""
    aa_idx = []
    for atom in traj_sel.topology.atoms:
        res = atom.residue.name.upper()
        aa_idx.append(AA_TO_IDX.get(res, AA_UNK))
    return torch.tensor(np.asarray(aa_idx, dtype=np.float32)[:, None], dtype=torch.float32)


# ----------------------------
# Bonds within selection: build edge_index + static bond meta
# ----------------------------
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
BOND_TO_IDX = {b: i for i, b in enumerate(BOND_TYPES)}

def _infer_bond_type_mdtraj(bond) -> str:
    """
    MDTraj bond objects sometimes have .order or .type depending on topology source.
    We try to infer; if unavailable return None.
    """
    # Common possibilities:
    # - bond.order (1,2,3)
    # - bond.type or bond.bond_type (string)
    # - bond.is_aromatic
    bt = None

    # aromatic flag
    if hasattr(bond, "is_aromatic") and bool(getattr(bond, "is_aromatic")):
        return "AROMATIC"

    # numeric order
    if hasattr(bond, "order"):
        try:
            o = int(getattr(bond, "order"))
            if o == 1: return "SINGLE"
            if o == 2: return "DOUBLE"
            if o == 3: return "TRIPLE"
        except Exception:
            pass

    # string type
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
    Returns:
      edge_index: (2, E)
      edge_pairs: (E, 2) int64, node indices i,j (for computing lengths)
      bond_type_oh: (E, 4) float32 onehot; zeros if unknown

    topology: mdtraj.Topology (FULL)
    atom_indices: selected atom indices in FULL topology
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

            # directed edge i->j
            edges.append((i, j))
            pairs.append((i, j))
            type_oh.append(oh)

            if make_undirected:
                edges.append((j, i))
                pairs.append((j, i))
                type_oh.append(oh)

    if len(edges) == 0:
        raise RuntimeError("No bonds found within selected atoms. Check selection/topology.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()          # (2,E)
    edge_pairs = torch.tensor(np.asarray(pairs), dtype=torch.long)              # (E,2)
    bond_type_oh = torch.tensor(np.asarray(type_oh), dtype=torch.float32)       # (E,4)

    return edge_index, edge_pairs, bond_type_oh


def compute_equilibrium_bond_lengths(traj_sel: md.Trajectory, edge_pairs: torch.Tensor) -> torch.Tensor:
    """
    Use the FIRST frame as "equilibrium" length (nm) for each directed edge.
    edge_pairs: (E,2) node indices i,j in selected indexing
    Returns: (E,1) eq_len_nm
    """
    xyz0 = traj_sel.xyz[0].astype(np.float32)  # (n_sel,3) nm
    ij = edge_pairs.cpu().numpy()
    vec = xyz0[ij[:, 0]] - xyz0[ij[:, 1]]
    d = np.sqrt((vec * vec).sum(axis=1, keepdims=True)).astype(np.float32)
    return torch.tensor(d, dtype=torch.float32)


def compute_current_bond_lengths(frame_sel_xyz_nm: np.ndarray, edge_pairs: torch.Tensor) -> torch.Tensor:
    """
    frame_sel_xyz_nm: (n_sel,3) float32
    edge_pairs: (E,2) torch long
    Returns: (E,1) cur_len_nm
    """
    ij = edge_pairs.cpu().numpy()
    vec = frame_sel_xyz_nm[ij[:, 0]] - frame_sel_xyz_nm[ij[:, 1]]
    d = np.sqrt((vec * vec).sum(axis=1, keepdims=True)).astype(np.float32)
    return torch.tensor(d, dtype=torch.float32)


# ----------------------------
# Graph per frame
# ----------------------------
def create_pyg_graph(
    traj_full: md.Trajectory,
    traj_sel: md.Trajectory,
    sel_atom_indices: np.ndarray,
    frame_idx: int,
    simulation: Simulation,
    property: str,
    atomic_number: torch.Tensor,      # (n_sel,1)
    residue_number: torch.Tensor,     # (n_sel,1)
    amino_acid_index: torch.Tensor,   # (n_sel,1)
    edge_index_sel: torch.Tensor,     # (2,E)
    edge_pairs: torch.Tensor,         # (E,2)
    bond_type_oh: torch.Tensor,       # (E,4)
    eq_len: torch.Tensor,             # (E,1)
) -> Data.Data:
    frame_full = traj_full[frame_idx]
    frame_sel = traj_sel[frame_idx]

    # positions for selected atoms (nm)
    pos_np = frame_sel.xyz[0].astype(np.float32)  # (n_sel,3)
    pos = torch.tensor(pos_np, dtype=torch.float32)

    # node features: (n_sel, 3)
    x = torch.cat([atomic_number, residue_number, amino_acid_index, pos], dim=1)

    # edge_attr dynamic lengths
    cur_len = compute_current_bond_lengths(pos_np, edge_pairs)          # (E,1)
    delta_len = cur_len - eq_len                                       # (E,1)

    # edge_attr: (E, 1+1+1+4 = 7)
    edge_attr = torch.cat([eq_len, cur_len, delta_len, bond_type_oh], dim=1)

    # energy from FULL system
    full_pos_nm = frame_full.xyz[0].astype(np.float32)  # (n_full,3)
    simulation.context.setPositions(full_pos_nm * nanometer)
    state = simulation.context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    # optional property on selected atoms
    timepoint = float(traj_full.time[frame_idx])
    if property == "rog":
        y = float(md.compute_rg(frame_sel)[0])
    elif property == "sasa":
        sasa = md.shrake_rupley(frame_sel, mode="atom")
        y = float(np.sum(sasa[0]))
    elif property in ("none", None):
        y = 0.0
    else:
        raise ValueError(f"Unknown property={property}. Use rog | sasa | none")

    graph = Data.Data(
        x=x,                       # (n_sel,3)
        pos=pos,                   # (n_sel,3) nm
        edge_index=edge_index_sel, # (2,E)
        edge_attr=edge_attr,       # (E,7)
        num_nodes=pos.shape[0],
        time=torch.tensor([timepoint], dtype=torch.float32),
        y=torch.tensor([y], dtype=torch.float32),
        energy=torch.tensor([potential_energy], dtype=torch.float32),
        sel_atom_indices=torch.tensor(sel_atom_indices, dtype=torch.long),
    )
    return graph


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare DE Shaw Ubiquitin trajectories as PyTorch Geometric graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_deshaw.py
  python prepare_deshaw.py --selection "protein and backbone" --property rog
  python prepare_deshaw.py --output Ubiquitin_graphs.pkl
        """
    )
    
    parser.add_argument('--selection', default="protein and backbone", help='MDTraj atom selection (default: "protein and backbone")')
    parser.add_argument('--property', default="rog", choices=["rog", "sasa", "none"], help='Node property to compute (default: rog)')
    parser.add_argument('--output', '-o', default="Ubiquitin_graphs.pkl", help='Output pickle file (default: Ubiquitin_graphs.pkl)')
    parser.add_argument('--datasets_dir', default="datasets/Ubiquitin", help='Ubiquitin dataset directory (default: datasets/Ubiquitin)')
    
    args = parser.parse_args()
    
    print(f"[info] Processing DE Shaw Ubiquitin dataset")
    print(f"[info] Selection: '{args.selection}'")
    print(f"[info] Property: {args.property}")
    
    # Find files in Ubiquitin dataset directory
    datasets_path = Path(args.datasets_dir)
    
    if not datasets_path.exists():
        print(f"[error] Directory not found: {datasets_path}")
        return 1
    
    # Look for trajectory and topology files
    xtc_files = list(datasets_path.glob("*.xtc"))
    pdb_files = list(datasets_path.glob("*.pdb"))
    
    if not xtc_files:
        print(f"[error] No .xtc file found in {datasets_path}")
        return 1
    if not pdb_files:
        print(f"[error] No .pdb file found in {datasets_path}")
        return 1
    
    # Use the trajectory and topology files
    xtc_path = str(xtc_files[0])
    pdb_path = str(pdb_files[0])
    
    print(f"[info] Found trajectory: {xtc_path}")
    print(f"[info] Found topology: {pdb_path}")
    
    # Setup and load
    try:
        simulation, _ = setup_simulation(pdb_path)
        traj_full = load_data(xtc_path, pdb_path)
    except Exception as e:
        print(f"[error] Failed to load files: {e}")
        return 1

    sel_atom_indices = traj_full.topology.select(args.selection)
    traj_sel = traj_full.atom_slice(sel_atom_indices)

    print(f"[info] full atoms={traj_full.topology.n_atoms}, selected atoms={traj_sel.topology.n_atoms}")
    print(f"[info] n_frames={traj_full.n_frames}")

    # edges + bond types (static)
    edge_index_sel, edge_pairs, bond_type_oh = build_bond_graph_for_subset(
        traj_full.topology,
        sel_atom_indices,
        make_undirected=True
    )
    print(f"[info] edge_index_sel shape={tuple(edge_index_sel.shape)}")
    print(f"[info] edge_attr static bond_type_oh shape={tuple(bond_type_oh.shape)}")

    # static node features
    atomic_number = atomic_number_feature(traj_sel)
    residue_number = residue_number_feature(traj_sel)      # RAW residue indices
    amino_acid_index = amino_acid_index_feature(traj_sel)

    # equilibrium bond lengths from first frame
    eq_len = compute_equilibrium_bond_lengths(traj_sel, edge_pairs)     # (E,1)

    graphs = []
    for frame_idx in tqdm(range(traj_full.n_frames), desc="Building graphs"):
        g = create_pyg_graph(
            traj_full=traj_full,
            traj_sel=traj_sel,
            sel_atom_indices=sel_atom_indices,
            frame_idx=frame_idx,
            simulation=simulation,
            property=args.property,
            atomic_number=atomic_number,
            residue_number=residue_number,
            amino_acid_index=amino_acid_index,
            edge_index_sel=edge_index_sel,
            edge_pairs=edge_pairs,
            bond_type_oh=bond_type_oh,
            eq_len=eq_len,
        )
        graphs.append(g)

    # Ensure graphs directory exists
    os.makedirs("graphs", exist_ok=True)
    output_path = os.path.join("graphs", args.output)
    with open(output_path, "wb") as f:
        pickle.dump(graphs, f)

    print(f"[done] wrote {args.output}")
    print(f"[dims] x_dim={graphs[0].x.shape} (expected [n,7])")
    print(f"[dims] pos_dim={graphs[0].pos.shape} (expected [n,3])")
    print(f"[dims] edge_attr_dim={graphs[0].edge_attr.shape} (expected [E,7])")
    print(f"[example] energy[0]={graphs[0].energy.item():.6f} kJ/mol")
    
    return 0


if __name__ == "__main__":
    exit(main())

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.dihedrals import Ramachandran

import numpy as np

def load_aligned_trajectories(
    top1: str,
    traj1: str | None,
    top2: str,
    traj2: str | None,
    align_to: str,
    align_sel: str,
) -> tuple[np.ndarray, np.ndarray]:


    if traj1 is None:
        u1 = mda.Universe(top1)
    else:
        u1 = mda.Universe(top1, traj1)

    if traj2 is None:
        u2 = mda.Universe(top2)
    else:
        u2 = mda.Universe(top2, traj2)


    if align_to is not None:
        ref = mda.Universe(align_to)
        align.AlignTraj(u1, ref, select=align_sel, in_memory=True).run()
        align.AlignTraj(u2, ref, select=align_sel, in_memory=True).run()
    else:
        # Align u2 onto u1's first frame as reference
        align.AlignTraj(u2, u1, select=align_sel, in_memory=True).run()
        align.AlignTraj(u1, u1, select=align_sel, in_memory=True).run()


    return u1, u2

def featurize_coordinates(u: mda.Universe, sel: str) -> np.ndarray:

    sel_atoms = u.select_atoms(sel)
    n_frames = len(u.trajectory)
    n_atoms = len(sel_atoms)

    coords = np.empty((n_frames, n_atoms, 3), dtype=float)

    for i, ts in enumerate(u.trajectory):
        frame_coords = sel_atoms.positions
        coords[i] = frame_coords

    return coords


def featurize_dihedrals(u: mda.Universe, embed: str = "sincos") -> np.ndarray:

    prot = u.select_atoms("protein")
    if prot.n_atoms == 0:
        raise ValueError("No protein atoms found with selection: 'protein'")

    rama = Ramachandran(prot).run()
    angles = rama.angles  # shape ~ (n_frames, n_res, 2) : phi, psi in radians

    # Replace NaNs (terminal residues often) by 0; alternatively you could mask residues.
    angles = np.nan_to_num(angles, nan=0.0)

    if embed == "raw":
        return angles
    elif embed == "sincos":
        phi = angles[:, :, 0]
        psi = angles[:, :, 1]
        X = np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=-1)  # (f,res,4)
        return X
    else:
        raise ValueError(f"Unknown dihedral embedding: {embed}")

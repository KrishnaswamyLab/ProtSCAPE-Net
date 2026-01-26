"""
What this script does (end-to-end):
1) Generates MULTIPLE latent-space paths in parallel (batched) using your mfd_ula_force_momentum.
2) Decodes each path to Cartesian xyz (nm).
3) Computes Onsager–Machlup (OM) action + relative path probability in DECODED space using OpenMM forces.
4) Visualizes all paths together on ONE PCA (and ONE PHATE) plot over the dataset latent cloud.
5) Optionally exports PDBs per-path and optionally runs MolProbity (via phenix.molprobity if available).

Notes:
- Path probability is NOT computed in latent space. It is computed after decoding to xyz, using OpenMM forces.
- If your model decodes a selection of atoms, enable --action_use_full_system so OM action uses full-system forces
  and then restricts to the selected atoms for the OM term (recommended).
"""

from __future__ import annotations

from glob import glob
import os
import pickle
import time
from typing import Callable, Optional, Any, List, Tuple, Dict

import numpy as np
import torch

from protscape.protscape import ProtSCAPE
from protscape.mfd_ld import mfd_ula_force_momentum
# from train_dae import load_or_train_dae

# Import helper functions and utilities from utils/
from utils.config import load_config, config_to_hparams
from utils.generation_helpers import (
    identity_denoiser,
    out_prefix_from_path,
    ensure_tensor_latents,
    load_module_from_path,
    pick_default_pdb,
    normalize_energy_inplace,
    get_energy_value,
    safe_abs,
    load_mu_sd,
)
from utils.openmm_eval import (
    OpenMMForceEvaluator,
    build_grad_fn_openmm,
)
from utils.generation_decode import (
    decode_paths_to_xyz_nm,
    compute_om_action_and_prob,
)
from utils.generation_viz import (
    plot_multi_paths_pca,
    plot_multi_paths_phate,
    export_pdb_frames,
    run_molprobity_on_folder,
)

# ============================================================================
# Main execution
# ============================================================================

def main():
    """
    Main generation script.
    
    Usage:
      python generation.py --config configs/config_generation.yaml
      
    Or with command-line overrides:
      python generation.py --config configs/config_generation.yaml --n_paths 16 --steps 20
    """
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config file (.yaml or .json)")
    
    # Allow command-line overrides
    cli_args, remaining = parser.parse_known_args()

    # Load config from file
    config = load_config(cli_args.config)
    
    # Convert to hparams namespace
    args = config_to_hparams(config)
    
    # Parse remaining args as overrides
    override_parser = ArgumentParser()
    for key, value in config.items():
        if isinstance(value, bool):
            override_parser.add_argument(f"--{key}", action="store_true")
        elif isinstance(value, int):
            override_parser.add_argument(f"--{key}", type=int)
        elif isinstance(value, float):
            override_parser.add_argument(f"--{key}", type=float)
        else:
            override_parser.add_argument(f"--{key}", type=str)
    
    override_args = override_parser.parse_args(remaining)
    
    # Apply overrides
    for key, value in vars(override_args).items():
        if value is not None:
            setattr(args, key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Dataset load (extend as needed)
        # Dataset load (extend as needed)
    protein_id = args.protein.lower()
    dataset_path = None
    
    # Try pattern: {protein_id}_{chain}_graphs.pkl
    pattern = f"*{protein_id}*graphs*.pkl"
    matches = glob.glob(pattern)
    
    if matches:
        dataset_path = matches[0]
        print(f"[dataset] auto-detected: {dataset_path}")
    else:
        raise FileNotFoundError(
            f"No dataset found matching pattern '{pattern}'. "
            f"Expected format: {{pdbid}}_{{chain}}_graphs.pkl"
        )

    with open(dataset_path, "rb") as f:
        full_dataset = pickle.load(f)

    with open(dataset_path, "rb") as f:
        full_dataset = pickle.load(f)
    if len(full_dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    # Normalize dataset energies for plotting / start selection
    energy_mu, energy_sd, base_energy_raw = normalize_energy_inplace(full_dataset)
    base_energy_norm = np.array([get_energy_value(g) for g in full_dataset], dtype=np.float64)
    print(f"[energy] normalized dataset energies with mu={energy_mu:.6f}, sd={energy_sd:.6f}")

    # Model arg setup (unchanged logic)
    args.num_nodes = full_dataset[0].x.shape[0]
    args.node_feat_dim = full_dataset[0].x.shape[1]
    args.input_dim = 3
    args.prot_graph_size = args.num_nodes

    Z_max = int(max(g.x[:, 0].max().item() for g in full_dataset))
    res_max = int(max(g.x[:, 1].max().item() for g in full_dataset))
    aa_max = int(max(g.x[:, 2].max().item() for g in full_dataset))
    print("[vocab]", Z_max, res_max, aa_max)

    args.num_Z = Z_max + 1
    args.num_residues = res_max + 1
    args.num_aa = max(aa_max + 1, 21)

    # Latents
    latents_all = np.load(args.latent_path, allow_pickle=True)
    latents_all = ensure_tensor_latents(latents_all)
    print(f"[latents] shape={tuple(latents_all.shape)}")

    # Base energies must align to latents length for coloring
    lat_all_np = latents_all.detach().cpu().numpy()
    if base_energy_norm.shape[0] != lat_all_np.shape[0]:
        print(
            f"[warn] base_energy length {base_energy_norm.shape[0]} != latents length {lat_all_np.shape[0]}; "
            "coloring may be misaligned. Will pad with NaNs."
        )
        be = np.full((lat_all_np.shape[0],), np.nan, dtype=np.float64)
        n = min(be.shape[0], base_energy_norm.shape[0])
        be[:n] = base_energy_norm[:n]
        base_energy_norm = be

    # Model load
    model = ProtSCAPE(args).to(device).eval()
    weights = torch.load(args.model_path, map_location=device)
    model.load_state_dict(weights)

    # OpenMM setup module
    atlas_rt = load_module_from_path(args.atlas_rt_path, module_name="atlas_new")
    if not hasattr(atlas_rt, "setup_simulation"):
        raise AttributeError(f"{args.atlas_rt_path} does not define setup_simulation")
    setup_simulation = getattr(atlas_rt, "setup_simulation")

    pdb_path = args.pdb or pick_default_pdb(args.protein)
    if pdb_path is None or not os.path.exists(pdb_path):
        raise FileNotFoundError("No PDB found. Provide --pdb or place a PDB in a known analysis folder.")
    print(f"[pdb] {pdb_path}")

    # Build latent gradient fn
    grad_fn = build_grad_fn_openmm(
        model=model,
        dataset=full_dataset,
        setup_simulation_fn=setup_simulation,
        pdb_path=pdb_path,
        device=device,
    )

    # DAE denoiser
    if args.use_dae:
        hidden_dims = [int(x.strip()) for x in args.dae_hidden_dims.split(",") if x.strip()]
        dae = load_or_train_dae(
            latent_path=args.latent_path,
            model_save_path=args.dae_model_path,
            latent_dim=args.latent_dim,
            hidden_dims=hidden_dims,
            n_epochs=args.dae_epochs,
            train_if_missing=args.dae_train_if_missing,
        )
        denoiser = dae
        print(f"[dae] enabled ({args.dae_model_path})")
    else:
        denoiser = identity_denoiser
        print("[dae] disabled")

    # ---------- choose MULTIPLE start points ----------
    valid = np.isfinite(base_energy_norm)
    energies = base_energy_norm.copy()
    energies[~valid] = np.nan
    rng = np.random.default_rng(0)

    if args.start_mode == "pctl":
        thr = np.nanpercentile(energies, float(args.start_percentile))
        order = np.argsort(np.abs(energies - thr))
        start_indices = [int(i) for i in order[: args.n_paths]]
    elif args.start_mode == "topk_random":
        thr = np.nanpercentile(energies, 80.0)
        pool = np.where(energies >= thr)[0]
        if pool.size < args.n_paths:
            raise RuntimeError(f"Not enough points in top 20% pool ({pool.size}) for n_paths={args.n_paths}")
        start_indices = [int(i) for i in rng.choice(pool, size=args.n_paths, replace=False)]
    elif args.start_mode == "same_point":
        start_idx = int(np.nanargmax(energies))
        start_indices = [start_idx for _ in range(args.n_paths)]
    else:
        pool = np.where(valid)[0]
        if pool.size < args.n_paths:
            raise RuntimeError(f"Not enough valid energy points ({pool.size}) for n_paths={args.n_paths}")
        start_indices = [int(i) for i in rng.choice(pool, size=args.n_paths, replace=False)]

    print("[starts]", start_indices)
    start = latents_all[start_indices].clone()  # (B,D) where B=n_paths
    if args.start_jitter > 0:
        start = start + float(args.start_jitter) * torch.randn_like(start)

    # ---------- generate MULTIPLE paths ----------
    print(f"[mfd_ula_force_momentum] steps={args.steps} step_size={args.step_size} n_paths={args.n_paths}")
    t0 = time.time()
    traj = mfd_ula_force_momentum(
        grad_fn,
        denoiser,
        start,
        args.steps,
        args.step_size,
        step_size_noise=1e-2,
        step_size_rescale=float(args.step_size_rescale),
        return_history=True,
        momentum=float(args.momentum),
    )
    t1 = time.time()
    print(f"[done] time={t1 - t0:.2f}s; traj shape={tuple(traj.shape)}")  # (T,B,D)

    # ---------- decode paths ----------
    xyz_paths_nm, t_idx = decode_paths_to_xyz_nm(
        model=model,
        traj=traj,
        device=device,
        decode_every=max(1, args.decode_every),
    )
    print(f"[decode] xyz_paths_nm shape={xyz_paths_nm.shape} (Tsub,B,N,3)")

    # If normalize_xyz was used in training: denorm decoded coords BEFORE exporting / action
    xyz_mu = xyz_sd = None
    if args.normalize_xyz:
        xyz_mu, xyz_sd = load_mu_sd(args)
        xyz_mu = np.asarray(xyz_mu, dtype=np.float64)
        xyz_sd = np.asarray(xyz_sd, dtype=np.float64)

        # Allow (N,3) or (1,N,3) broadcasting
        if xyz_mu.ndim == 2:
            xyz_mu = xyz_mu[None, ...]
        if xyz_sd.ndim == 2:
            xyz_sd = xyz_sd[None, ...]
        # Apply: xyz = xyz*sd + mu  (nm)
        xyz_paths_nm = xyz_paths_nm * xyz_sd + xyz_mu
        print("[decode] applied denorm: xyz = xyz*sd + mu")

    # Save decoded xyz
    prefix = out_prefix_from_path(args.out)
    np.save(prefix + "_decoded_xyz_paths_nm.npy", xyz_paths_nm)
    print(f"[saved] decoded xyz paths (nm): {safe_abs(prefix + '_decoded_xyz_paths_nm.npy')}")

    # ---------- OM action + relative path probability in DECODED space ----------
    force_eval = OpenMMForceEvaluator(setup_simulation, pdb_path=pdb_path, first_data=full_dataset[0])

    sel_idx = None
    if hasattr(full_dataset[0], "sel_atom_indices"):
        sel_idx = full_dataset[0].sel_atom_indices.detach().cpu().numpy().astype(int)

    actions, rel_prob = compute_om_action_and_prob(
        xyz_paths_nm=xyz_paths_nm,                   # (Tsub,B,N,3)
        force_eval=force_eval,
        pdb_path=pdb_path,
        use_full_system=bool(args.action_use_full_system),
        sel_idx=sel_idx,
        dt=float(args.action_dt),
    )

    for b in range(args.n_paths):
        print(f"[path {b:02d}] OM_action={actions[b]:.6e}  rel_prob={rel_prob[b]:.4f}")
    best = int(np.argmax(rel_prob))
    print(f"[best] path={best:02d}  rel_prob={rel_prob[best]:.4f}  action={actions[best]:.6e}")

    # ---------- multipath PCA + PHATE plots ----------
    pca_path = prefix + "_pca_multipath.png"
    plot_multi_paths_pca(
        lat_all=lat_all_np,
        base_energy_norm=base_energy_norm,
        traj=traj,
        out_path=pca_path,
        title=f"PCA multipath: {args.protein} (best={best})",
        plot_every=max(1, args.plot_every),
    )
    print(f"[saved] PCA multipath: {safe_abs(pca_path)}")

    phate_path = prefix + "_phate_multipath.png"
    plot_multi_paths_phate(
        lat_all=lat_all_np,
        base_energy_norm=base_energy_norm,
        traj=traj,
        out_path=phate_path,
        title=f"PHATE multipath: {args.protein} (best={best})",
        knn=int(args.phate_knn),
        t=int(args.phate_t),
        plot_every=max(1, args.plot_every),
    )
    print(f"[saved] PHATE multipath: {safe_abs(phate_path)}")

    # ---------- export PDBs per path ----------
    molprobity_results: List[Dict[str, Any]] = []
    if args.export_pdb:
        base_dir = prefix + "_pdb_paths"
        os.makedirs(base_dir, exist_ok=True)

        # per-path folders
        for b in range(args.n_paths):
            pdb_dir_b = os.path.join(base_dir, f"path_{b:02d}")
            abs_dir = export_pdb_frames(
                xyz_nm=xyz_paths_nm[:, b],  # (Tsub,N,3)
                pdb_ref_path=pdb_path,
                out_dir=pdb_dir_b,
                normalize_xyz=False,        # already denormed if needed
                xyz_mu=None,
                xyz_sd=None,
                n_pdb_frames=min(int(args.n_pdb_frames), xyz_paths_nm.shape[0]),
                align_kabsch=True,
                sel_atom_indices=sel_idx,
            )
            print(f"[saved] path {b:02d} PDBs: {abs_dir}")

            if args.run_molprobity:
                mp = run_molprobity_on_folder(pdb_dir_b)
                if mp:
                    mp["path"] = b
                    mp["om_action"] = float(actions[b])
                    mp["om_rel_prob"] = float(rel_prob[b])
                    molprobity_results.append(mp)

    # ---------- save payload ----------
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload: Dict[str, Any] = {
        "traj": traj,
        "xyz_paths_nm": None,  # keep file small; use the .npy saved path instead
        "decoded_xyz_paths_nm_path": safe_abs(prefix + "_decoded_xyz_paths_nm.npy"),
        "protein": args.protein,
        "pdb_path": pdb_path,
        "latent_path": args.latent_path,
        "model_path": args.model_path,
        "steps": int(args.steps),
        "step_size": float(args.step_size),
        "start_indices": start_indices,
        "n_paths": int(args.n_paths),
        "use_dae": bool(args.use_dae),
        "dae_model_path": args.dae_model_path if args.use_dae else None,
        "energy_mu": float(energy_mu),
        "energy_sd": float(energy_sd),

        # OM action + relative probability
        "om_action": actions,
        "om_rel_prob": rel_prob,
        "action_dt": float(args.action_dt),
        "action_use_full_system": bool(args.action_use_full_system),
        "best_path": int(best),

        # MolProbity (optional)
        "molprobity": molprobity_results,

        # plots
        "pca_plot_path": safe_abs(pca_path),
        "phate_plot_path": safe_abs(phate_path),
    }

    with open(args.out, "wb") as outfh:
        pickle.dump(payload, outfh)
    print(f"[saved] payload pkl: {safe_abs(args.out)}")

    # print quick summary table for copy/paste into notes
    print("\n=== SUMMARY ===")
    for b in range(args.n_paths):
        print(f"path {b:02d}: rel_prob={rel_prob[b]:.4f}  action={actions[b]:.6e}")
    if molprobity_results:
        print("\n=== MOLPROBITY (mean over frames) ===")
        for r in molprobity_results:
            print(
                f"path {int(r['path']):02d}: "
                f"clash={r.get('clashscore_mean', np.nan):.3f}  "
                f"rama_favored={r.get('rama_favored_mean', np.nan):.2f}%  "
                f"rama_outliers={r.get('rama_outliers_mean', np.nan):.2f}%  "
                f"rel_prob={r.get('om_rel_prob', np.nan):.4f}"
            )


if __name__ == "__main__":
    main()
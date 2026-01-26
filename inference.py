"""
Inference script for atomic-level ProtSCAPE models.
Atomic-level testing for graphs where:
  - nodes = atoms in a fixed selection (must match graph construction)
  - node features x = [Z, res_idx, aa_idx, xyz(3)]  (xyz in nm if made by MDTraj)
  - model outputs (NEW):
      z_rep: latent embedding (B, latent_dim)
      x_recon = (Z_logits, res_logits, aa_logits, xyz_pred)
      x_gt_dense, node_mask

Fixes PDB export units:
  - MDTraj xyz is in **nm**
  - MDAnalysis AtomGroup.positions expects **Å**
So we convert nm -> Å by multiplying by 10.0 before writing PDB (and before RMSD).

Adds:
  - saves latent representations (z_rep) + energies to disk
  - plots PCA and PHATE embeddings colored by energy

Outputs (in --out_dir):
  - pdb_frames/ (pred/true PDBs)
  - latents_zrep.npy, energies.npy, times.npy (if available)
  - pca_energy.png, phate_energy.png (if PHATE installed)
  - latent_raw_3d_energy.png (dims 0-2) if latent_dim >= 3

Notes:
  - For correct PDB export, graph[0].sel_atom_indices must exist and match MDAnalysis ordering.
  - Use --batch_size 1 if exporting PDBs.

Usage:
  python inference.py --config config_inference.yaml --ckpt_path path/to/model.pt
"""

from argparse import ArgumentParser
import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import MDAnalysis as mda
from protscape.protscape import ProtSCAPE
from utils.geometry import kabsch_rmsd, kabsch_align_np, mse_xyz
from utils.normalize import apply_xyz_norm, compute_xyz_norm_stats, normalize_energy
from utils.visualizations import compute_pca_2d, try_compute_phate_2d, plot_embedding
from utils.split import ensure_dir
from utils.config import load_config, config_to_hparams, save_config

NM_TO_ANG = 10.0

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config file (.yaml or .json)")

    cli_args = parser.parse_args()

    # Load config from file
    config = load_config(cli_args.config)
    
    # Convert to hparams namespace
    args = config_to_hparams(config)
    
    if args.ckpt_path is None:
        raise ValueError("ckpt_path must be provided in config or via --ckpt_path")


    ensure_dir(args.out_dir)
    pdb_dir = os.path.join(args.out_dir, "pdb_frames")
    ensure_dir(pdb_dir)

    # -----------------------------
    # Load dataset
    # -----------------------------
    with open(args.graphs_pkl, "rb") as f:
        full_dataset = pickle.load(f)

    for i, g in enumerate(full_dataset):
        g.time = int(getattr(g, "time", i))

    print(f"[info] Loaded {len(full_dataset)} graphs from {args.graphs_pkl}")
    print(f"[info] Example x shape: {full_dataset[0].x.shape} (nodes, feat_dim)")

    # infer dims
    x0 = full_dataset[0].x
    if not isinstance(x0, torch.Tensor):
        x0 = torch.tensor(np.asarray(x0), dtype=torch.float32)

    Z_max = int(max(g.x[:,0].max().item() for g in full_dataset))
    res_max = int(max(g.x[:,1].max().item() for g in full_dataset))
    aa_max = int(max(g.x[:,2].max().item() for g in full_dataset))
    print(Z_max, res_max, aa_max)

    args.num_Z = Z_max + 1
    args.num_residues = res_max + 1
    args.num_aa = max(aa_max + 1, 21)  # usually 21
    num_nodes0 = int(x0.shape[0])
    feat_dim0 = int(x0.shape[1])

    if feat_dim0 != 6:
        raise ValueError(f"Expected x feature dim = 6 for [Z,res_idx,aa_idx,xyz(3)]. Got {feat_dim0}.")

    # -----------------------------
    # Normalize energy + xyz dims only (optional)
    # -----------------------------
    if args.normalize_energy:
        energy_mean, energy_std = normalize_energy(full_dataset)
        print(f"[norm] energy mean={energy_mean:.6f}, std={energy_std:.6f}")

    if args.normalize_xyz:
        xyz_mu, xyz_sd = compute_xyz_norm_stats(full_dataset, xyz_start=3)
        apply_xyz_norm(full_dataset, xyz_mu, xyz_sd, xyz_start=3)
        print(f"[norm] xyz mean={xyz_mu.squeeze()}, std={xyz_sd.squeeze()}")
    else:
        xyz_mu, xyz_sd = None, None

    # -----------------------------
    # DataLoader
    # -----------------------------
    loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # -----------------------------
    # Build model args from data
    # -----------------------------
    args.residue_num = num_nodes0
    args.num_nodes = num_nodes0
    args.node_feat_dim = feat_dim0
    args.prot_graph_size = args.num_nodes
    args.len_epoch = len(loader)

    print(
        f"[dims] num_nodes={args.num_nodes}, node_feat_dim={args.node_feat_dim}, "
        f"prot_graph_size={args.prot_graph_size}, input_dim={args.input_dim}, "
        f"num_Z={args.num_Z}, num_residues={args.num_residues}, num_aa={args.num_aa}"
    )

    # -----------------------------
    # Load model weights
    # -----------------------------
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    model = ProtSCAPE(args)
    state = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # -----------------------------
    # Setup MDAnalysis for PDB export
    # -----------------------------
    if args.export_pdb:
        if args.xtc is not None and os.path.exists(args.xtc):
            u = mda.Universe(args.pdb, args.xtc)
        else:
            u = mda.Universe(args.pdb)

        if not hasattr(full_dataset[0], "sel_atom_indices"):
            raise RuntimeError(
                "Dataset graphs do not have sel_atom_indices. "
                "Please store sel_atom_indices during preprocessing so export ordering matches."
            )

        sel_idx = full_dataset[0].sel_atom_indices
        if torch.is_tensor(sel_idx):
            sel_idx = sel_idx.cpu().numpy()
        sel_idx = sel_idx.astype(int)

        ag = u.atoms[sel_idx]
        print(f"[info] Using stored sel_atom_indices for export: {len(ag)} atoms")

        if len(ag) != args.num_nodes:
            print(f"[warn] MDAnalysis AtomGroup has {len(ag)} atoms but graphs have {args.num_nodes} nodes.")
            print("       Export will be skipped unless counts match.")
        else:
            print(f"[info] MDAnalysis selection atoms = {len(ag)} (matches graph nodes).")
    else:
        ag = None

    # -----------------------------
    # Inference + metrics + latents
    # -----------------------------
    Z_acc_list = []
    res_acc_list = []
    aa_acc_list = []

    xyz_mse_norm_list = []
    xyz_mae_norm_list = []
    kabsch_rmsd_list = []
    kabsch_mse_list = []

    Z_all = []
    E_all = []
    T_all = []

    with torch.no_grad():
        for t, batch in enumerate(tqdm(loader, desc="Testing")):
            batch = batch.to(device)

            # Expected forward signature:
            # z_rep, coeffs, att_maps, x_recon, x_gt_dense, node_mask
            out = model(batch)

            if not (isinstance(out, (tuple, list)) and len(out) == 6):
                raise ValueError(
                    f"Expected model(batch) to return 6 items, got {type(out)} "
                    f"len={len(out) if isinstance(out,(tuple,list)) else 'NA'}"
                )

            z_rep, coeffs, att_maps, x_recon, x_gt_dense, node_mask = out

            # NEW recon:
            # x_recon = (Z_logits, res_logits, aa_logits, xyz_pred)
            Z_logits, res_logits, aa_logits, xyz_pred = x_recon

            # Targets: x_gt_dense = (B,N,6) with [Z, res_idx, aa_idx, xyz(3)]
            Z_gt = x_gt_dense[:, :, 0].round().long()
            res_gt = x_gt_dense[:, :, 1].round().long()
            aa_gt = x_gt_dense[:, :, 2].round().long()
            xyz_gt = x_gt_dense[:, :, 3:6]

            mask = node_mask
            mask_flat = mask.reshape(-1)

            # ----------------
            # Collect latents + energies
            # ----------------
            if args.save_latents or args.plot_latents:
                Z_all.append(z_rep.detach().cpu().numpy())

                if hasattr(batch, "energy"):
                    e = batch.energy
                    if torch.is_tensor(e):
                        e = e.detach().cpu().numpy()
                    e = np.asarray(e).reshape(-1)
                else:
                    e = np.full((z_rep.shape[0],), np.nan, dtype=np.float32)
                E_all.append(e)

                if hasattr(batch, "time"):
                    tt = batch.time
                    if torch.is_tensor(tt):
                        tt = tt.detach().cpu().numpy()
                    tt = np.asarray(tt).reshape(-1)
                else:
                    tt = np.full((z_rep.shape[0],), np.nan, dtype=np.float32)
                T_all.append(tt)

            # ----------------
            # Feature reconstruction accuracies
            # ----------------
            Z_pred = Z_logits.argmax(dim=-1).reshape(-1)
            res_pred = res_logits.argmax(dim=-1).reshape(-1)
            aa_pred = aa_logits.argmax(dim=-1).reshape(-1)

            Z_true = Z_gt.reshape(-1).clamp(0, args.num_Z - 1)
            res_true = res_gt.reshape(-1).clamp(0, args.num_residues - 1)
            aa_true = aa_gt.reshape(-1).clamp(0, args.num_aa - 1)

            if mask_flat.any():
                Z_acc = (Z_pred[mask_flat] == Z_true[mask_flat]).float().mean().item()
                res_acc = (res_pred[mask_flat] == res_true[mask_flat]).float().mean().item()
                aa_acc = (aa_pred[mask_flat] == aa_true[mask_flat]).float().mean().item()
            else:
                Z_acc = res_acc = aa_acc = float("nan")

            Z_acc_list.append(Z_acc)
            res_acc_list.append(res_acc)
            aa_acc_list.append(aa_acc)

            # ----------------
            # XYZ metrics in feature space (nm or normalized-nm)
            # ----------------
            xyz_pred_flat = xyz_pred.reshape(-1, 3)
            xyz_gt_flat = xyz_gt.reshape(-1, 3)
            if mask_flat.any():
                xyz_pred_flat = xyz_pred_flat[mask_flat]
                xyz_gt_flat = xyz_gt_flat[mask_flat]
                xyz_mse_norm_list.append(F.mse_loss(xyz_pred_flat, xyz_gt_flat).item())
                xyz_mae_norm_list.append(F.l1_loss(xyz_pred_flat, xyz_gt_flat).item())
            else:
                xyz_mse_norm_list.append(float("nan"))
                xyz_mae_norm_list.append(float("nan"))

            # ----------------
            # De-normalize XYZ for physical-space eval/export (nm), then convert to Å
            # ----------------
            if (args.export_pdb and t < args.n_pdb_frames) and xyz_pred.shape[0] != 1:
                raise ValueError("For --export_pdb please use --batch_size 1.")

            xyz_pred_np = xyz_pred[0].detach().cpu().numpy()
            xyz_true_np = xyz_gt[0].detach().cpu().numpy()

            if args.normalize_xyz:
                xyz_pred_np = xyz_pred_np * xyz_sd + xyz_mu
                xyz_true_np = xyz_true_np * xyz_sd + xyz_mu

            xyz_pred_A = (xyz_pred_np * NM_TO_ANG).astype(np.float64)
            xyz_true_A = (xyz_true_np * NM_TO_ANG).astype(np.float64)

            # Kabsch metrics in Å
            try:
                rmsd = kabsch_rmsd(xyz_pred_A, xyz_true_A)
                kabsch_rmsd_list.append(rmsd)

                xyz_pred_aligned_A, _ = kabsch_align_np(xyz_pred_A, xyz_true_A)
                kabsch_mse_list.append(mse_xyz(xyz_pred_aligned_A, xyz_true_A))
            except Exception:
                xyz_pred_aligned_A = xyz_pred_A
                kabsch_rmsd_list.append(float("nan"))
                kabsch_mse_list.append(float("nan"))

            # Export PDBs (Å). Write ALIGNED pred so it overlays nicely.
            if args.export_pdb and t < args.n_pdb_frames and ag is not None:
                if len(ag) == xyz_pred_aligned_A.shape[0]:
                    ag.positions = xyz_pred_aligned_A.astype(np.float32)
                    ag.write(os.path.join(pdb_dir, f"pred_frame_{t:05d}.pdb"))
                    ag.positions = xyz_true_A.astype(np.float32)
                    ag.write(os.path.join(pdb_dir, f"true_frame_{t:05d}.pdb"))
                else:
                    print(
                        f"[warn] Skipping PDB export at frame {t}: "
                        f"AtomGroup has {len(ag)} atoms but xyz has {xyz_pred_A.shape[0]}."
                    )

    # -----------------------------
    # Save latents and plots
    # -----------------------------
    if args.save_latents or args.plot_latents:
        Z = np.concatenate(Z_all, axis=0) if len(Z_all) else np.zeros((0, args.latent_dim), dtype=np.float32)
        E = np.concatenate(E_all, axis=0) if len(E_all) else np.zeros((0,), dtype=np.float32)
        TT = np.concatenate(T_all, axis=0) if len(T_all) else np.zeros((0,), dtype=np.float32)

        np.save(os.path.join(args.out_dir, "latents_zrep_10k.npy"), Z.astype(np.float32))
        np.save(os.path.join(args.out_dir, "energies_10k.npy"), E.astype(np.float32))
        np.save(os.path.join(args.out_dir, "times_10k.npy"), TT.astype(np.float32))
        print(f"[latents] Saved: {args.out_dir}/latents_zrep_10k.npy (shape={Z.shape})")

        if args.plot_latents and Z.shape[0] > 1:
            color = E.copy()

            # PCA
            pca2 = compute_pca_2d(Z)
            plot_embedding(
                pca2, color,
                out_path=os.path.join(args.out_dir, "pca_energy_10k.png"),
                title="PCA(z_rep) colored by energy",
            )
            print(f"[latents] Wrote {args.out_dir}/pca_energy_10k.png")

            # Raw latent dims 0-2 in 3D (if available)
            if Z.shape[1] >= 3:
                import matplotlib.pyplot as plt

                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")

                scatter = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=color, s=20, alpha=0.6)
                ax.set_xlabel("Latent Dim 0")
                ax.set_ylabel("Latent Dim 1")
                ax.set_zlabel("Latent Dim 2")
                ax.set_title("Raw Latent Space (dims 0-2) colored by energy")

                plt.colorbar(scatter, ax=ax, label="Energy", shrink=0.8)
                plt.tight_layout()
                plt.savefig(os.path.join(args.out_dir, "latent_raw_3d_energy_10k.png"), dpi=200)
                plt.close()
                print(f"[latents] Wrote {args.out_dir}/latent_raw_3d_energy_10k.png")

            # PHATE
            ph2 = try_compute_phate_2d(Z, seed=args.seed, knn=args.phate_knn, t=args.phate_t)
            if ph2 is None:
                print("[latents] PHATE not installed; skipping PHATE plot. (pip install phate)")
            else:
                plot_embedding(
                    ph2, color,
                    out_path=os.path.join(args.out_dir, "phate_energy_10k.png"),
                    title="PHATE(z_rep) colored by energy",
                )
                print(f"[latents] Wrote {args.out_dir}/phate_energy_10k.png")

    # -----------------------------
    # Print summary
    # -----------------------------
    print("\n===== Feature Reconstruction Quality =====")
    print(f"Z (atomic number) accuracy: mean={np.nanmean(Z_acc_list):.4f} std={np.nanstd(Z_acc_list):.4f}")
    print(f"Residue index accuracy:     mean={np.nanmean(res_acc_list):.4f} std={np.nanstd(res_acc_list):.4f}")
    print(f"Amino acid index accuracy:  mean={np.nanmean(aa_acc_list):.4f} std={np.nanstd(aa_acc_list):.4f}")

    print("\n===== XYZ Reconstruction Quality =====")
    print(f"XYZ MSE (feature space): mean={np.nanmean(xyz_mse_norm_list):.6f} std={np.nanstd(xyz_mse_norm_list):.6f}")
    print(f"XYZ MAE (feature space): mean={np.nanmean(xyz_mae_norm_list):.6f} std={np.nanstd(xyz_mae_norm_list):.6f}")

    print("\n===== Physical-space Geometry (Å, de-normalized xyz) =====")
    print(f"RMSD (Kabsch, Å): mean={np.nanmean(kabsch_rmsd_list):.4f} std={np.nanstd(kabsch_rmsd_list):.4f}")
    print(f"Kabsch-aligned XYZ MSE (Å^2): mean={np.nanmean(kabsch_mse_list):.6f} std={np.nanstd(kabsch_mse_list):.6f}")

    if args.export_pdb:
        print(f"\nExported atomic PDB frames to: {pdb_dir}")
        print("PyMOL tip:")
        print(f"  cd {pdb_dir}")
        print("  load pred_frame_00000.pdb")
        print("  load true_frame_00000.pdb")
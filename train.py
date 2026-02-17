"""
Train ProtSCAPE on atomic-level graphs where node features are:
  x = [atom_onehot (atom_type_dim) | xyz (3)]

Key points:
- We can z-score normalize ENERGY across dataset (optional).
- We can z-score normalize ONLY the XYZ block (last 3 dims) across dataset (recommended).
  We DO NOT normalize the one-hot atom type block (or CrossEntropy breaks).
- Windowed interpolation split for train/val.
- Model uses SE(3)/E(3)-equivariant MP (Clof_GCL + E_GCL) internally.
- Coordinate reconstruction loss is TorchMetrics Procrustes disparity (scale-invariant).

Usage:
  python train.py --config config.yaml
  
  Or with override:
  python train.py --config config.yaml --batch_size 32 --n_epochs 500
"""

from argparse import ArgumentParser
import datetime
import os
import pickle
import warnings
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from protscape.protscape import ProtSCAPE
from utils.normalize import normalize_energy, normalize_xyz_only
from utils.split import windowed_interpolation_split
from utils.config import load_config, config_to_hparams, save_config

# Suppress all warnings
warnings.filterwarnings("ignore")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config file (.yaml or .json)")
    parser.add_argument("--protein", type=str, help="Protein ID (e.g., '7lp1', '1bx7', '7jfl') to override config")
    parser.add_argument("--pkl_path", type=str, help="Path to pickle file to override config")
    
    cli_args = parser.parse_args()

    # Load config from file
    config = load_config(cli_args.config)
    
    # Override config with command-line arguments if provided
    if cli_args.protein:
        config["protein"] = cli_args.protein
        # Auto-generate pkl_path if not explicitly provided
        if not cli_args.pkl_path:
            protein_prefix = cli_args.protein.split("_")[0]  # e.g., "7lp1" from "7lp1_A"
            config["pkl_path"] = f"data/graphs/{cli_args.protein}_C_graphs.pkl"
    
    if cli_args.pkl_path:
        config["pkl_path"] = cli_args.pkl_path
    
    # Convert to hparams namespace
    args = config_to_hparams(config)

    # -------------------------
    # Load graphs
    # -------------------------
    with open(args.pkl_path, "rb") as f:
        full_dataset = pickle.load(f)

    if not isinstance(full_dataset, (list, tuple)) or len(full_dataset) == 0:
        raise RuntimeError(f"Loaded empty dataset from {args.pkl_path}")

    print(f"Loaded {len(full_dataset)} graphs from {args.pkl_path}")
    # -------------------------
    # Infer dims + sanity checks
    # -------------------------
    x0 = full_dataset[0].x
    if not isinstance(x0, torch.Tensor):
        x0 = torch.tensor(np.asarray(x0), dtype=torch.float32)

    Z_max = int(max(g.x[:,0].max().item() for g in full_dataset))
    res_max = int(max(g.x[:,1].max().item() for g in full_dataset))
    aa_max = int(max(g.x[:,2].max().item() for g in full_dataset))
    args.num_Z = Z_max + 1
    args.num_residues = res_max + 1
    args.num_aa = max(aa_max + 1, 21)  # usually 21

    feat_dim0 = int(x0.shape[1])
    if feat_dim0 <= 3:
        raise ValueError(f"Feature dim F={feat_dim0} must be > 3 (needs onehot + xyz).")

    # assume last 3 are xyz
    args.atom_type_dim = 3
    args.node_feat_dim = int(x0.shape[1])
    args.num_nodes = int(x0.shape[0])
    # scattering/positional assumes max_seq_len >= num_nodes (fixed-size graphs)
    args.prot_graph_size = args.num_nodes

    # quick check: fixed node counts strongly recommended
    n0 = int(full_dataset[0].x.shape[0])
    if not all(int(g.x.shape[0]) == n0 for g in full_dataset):
        raise ValueError(
            "Node count differs across graphs. Keep atom selection fixed so all graphs have same #atoms.\n"
            "If you must allow variable nodes, we can rely on masking everywhere, but ensure Scatter_layer supports it."
        )
    # quick check: edge_index exists (bond edges)
    if not hasattr(full_dataset[0], "edge_index"):
        raise ValueError("Graphs must have edge_index (bond edges).")
    print(
        f"[dims] num_nodes={args.num_nodes}, node_feat_dim={args.node_feat_dim}, "
        f"atom_type_dim={args.atom_type_dim}, xyz_dim=3, prot_graph_size={args.prot_graph_size}, "
        f"input_dim(scatter/mp)={args.input_dim}"
    )
    # -------------------------
    # Normalize (energy + XYZ only)
    # -------------------------
    if args.normalize_x:
        xyz_mean, xyz_std = normalize_xyz_only(
            full_dataset,
            atom_type_dim=args.atom_type_dim,
            xyz_dim=3
        )
        args.xyz_mean = xyz_mean
        args.xyz_std = xyz_std
        print(f"[norm] xyz mean={xyz_mean}, std={xyz_std}")
    if args.normalize_energy:
        energy_mean, energy_std = normalize_energy(full_dataset)
        args.energy_mean = energy_mean
        args.energy_std = energy_std
        print(f"[norm] energy mean={energy_mean:.6f}, std={energy_std:.6f}")

    # -------------------------
    # Split
    # -------------------------
    train_set, val_set, full_dataset = windowed_interpolation_split(
        full_dataset,
        window_size=args.window_size,
        num_windows=args.num_windows,
        seed=args.split_seed,
    )

    print(f"Total number of graphs={len(full_dataset)}")

    train_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    args.len_epoch = len(train_loader)

    # -------------------------
    # Logging
    # -------------------------
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, f"progsnn_logs_run_{args.dataset}_{date_suffix}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the config used for this run (for reproducibility)
    save_config(config, os.path.join(save_dir, "config_used.yaml"))

    run_name = f"{args.dataset}_{args.protein}_ATOMIC_onehotXYZ_SE3_procrustes"
    wandb_logger = WandbLogger(
        name=run_name,
        project=args.wandb_project,
        log_model=True,
        save_dir=save_dir,
    )
    wandb_logger.log_hyperparams(vars(args))
    wandb_logger.experiment.log({"logging_timestamp": date_suffix})

    # -------------------------
    # Train
    # -------------------------
    model = ProtSCAPE(args)

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        devices="auto",
        logger=wandb_logger,
        # log_every_n_steps=10,
        # enable_checkpointing=True,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # -------------------------
    # Save
    # -------------------------
    model = model.cpu()
    out_path = os.path.join(save_dir, f"model_FINAL_{args.protein}.pt")
    print(f"[save] saving model -> {out_path}")
    torch.save(model.state_dict(), out_path)

    print("Done.")
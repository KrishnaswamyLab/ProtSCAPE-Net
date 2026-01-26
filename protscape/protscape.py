"""
ProtSCAPE model for protein structure reconstruction from ATLAS graphs.
Implements:
- EGNN layers for message passing
- Scattering transform layer
- Row-wise Transformer encoder
- Bottleneck module
- Decoders for atomic number, residue index, amino acid type, and 3D coordinates
- Kabsch alignment and loss for coordinate reconstruction
"""

import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from torch.nn.utils import spectral_norm

from egnn_pytorch import EGNN  # pip install egnn-pytorch

from protscape.bottleneck import BaseBottleneck
from protscape.transformer import PositionalEncoding, TransformerEncoder
from protscape.base import TGTransformerBaseModel_ATLAS
from protscape.wavelets import Scatter_layer


# -------------------------
# Kabsch alignment + loss
# -------------------------
def kabsch_align(pred_xyz, true_xyz, mask, eps: float = 1e-8, allow_reflection: bool = False):
    mask_f = mask.to(pred_xyz.dtype).unsqueeze(-1)  # (B,N,1)
    n_valid = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1,1)

    pred_centroid = (pred_xyz * mask_f).sum(dim=1, keepdim=True) / n_valid
    true_centroid = (true_xyz * mask_f).sum(dim=1, keepdim=True) / n_valid

    P = (pred_xyz - pred_centroid) * mask_f
    Q = (true_xyz - true_centroid) * mask_f

    H = P.transpose(1, 2) @ Q
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)

    R = V @ U.transpose(-2, -1)

    if not allow_reflection:
        detR = torch.det(R)
        neg = detR < 0
        if neg.any():
            V_fix = V.clone()
            V_fix[neg, :, 2] *= -1.0
            R = V_fix @ U.transpose(-2, -1)

    pred_aligned = (pred_xyz - pred_centroid) @ R + true_centroid
    return pred_aligned


def kabsch_mse_loss(pred_xyz, true_xyz, mask, allow_reflection: bool = False):
    pred_aligned = kabsch_align(pred_xyz, true_xyz, mask, allow_reflection=allow_reflection)
    diff2 = ((pred_aligned - true_xyz) ** 2).sum(dim=-1)  # (B,N)
    diff2 = diff2 * mask.to(diff2.dtype)
    denom = mask.to(diff2.dtype).sum().clamp_min(1.0)
    return diff2.sum() / denom


def _scalarize(x: torch.Tensor) -> torch.Tensor:
    return x.mean() if (isinstance(x, torch.Tensor) and x.ndim > 0) else x


# -------------------------
# helper: sparse PyG edges -> dense (B, N, N, edge_dim)
# -------------------------
def pyg_edges_to_dense_edges(edge_index, edge_attr, batch_vec, num_graphs, num_nodes, edge_dim, device):
    edges_dense = torch.zeros((num_graphs, num_nodes, num_nodes, edge_dim),
                             device=device, dtype=edge_attr.dtype)
    row, col = edge_index[0], edge_index[1]
    g = batch_vec[row]
    row_local = row - g * num_nodes
    col_local = col - g * num_nodes
    edges_dense[g, row_local, col_local] = edge_attr
    return edges_dense


class ProtSCAPE(TGTransformerBaseModel_ATLAS):
    """
    Expects batch.x per node:
      x[:,0] = atomic_number (Z)        (integer-like)
      x[:,1] = residue_index            (integer-like, 0..n_res-1)
      x[:,2] = amino_acid_index         (integer-like, 0..20)
      x[:,3:6] = xyz (nm)

    Reconstruction:
      - predict logits for Z / residue / aa (3 CE losses)
      - predict xyz with Kabsch MSE
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()

        self.input_dim = hparams.input_dim
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim
        self.max_seq_len = hparams.prot_graph_size

        self.layers = hparams.layers
        self.probs = hparams.probs
        self.nhead = hparams.nhead
        self.lr = hparams.lr

        self.alpha = getattr(hparams, "alpha", 1.0)
        self.beta_loss = getattr(hparams, "beta_loss", 1.0)
        self.coord_weight = getattr(hparams, "coord_weight", 1.0)
        self.allow_reflection = getattr(hparams, "allow_reflection", False)
        self.batch_size = getattr(hparams, "batch_size", 1)

        self.num_nodes = getattr(hparams, "num_nodes", None)
        if self.num_nodes is None:
            raise ValueError("hparams.num_nodes must be set.")

        self.node_feat_dim = getattr(hparams, "node_feat_dim", None)
        if self.node_feat_dim is None:
            raise ValueError("hparams.node_feat_dim must be set.")
        if self.node_feat_dim != 6:
            raise ValueError(f"Expected node_feat_dim=6 for [Z,res,aa,xyz]. Got {self.node_feat_dim}")

        # vocab sizes (YOU SHOULD SET THESE FROM DATASET STATS)
        # - atomic numbers: max_Z_seen+1 is ideal
        self.num_Z = getattr(hparams, "num_Z", 128)            # safe default
        self.num_residues = getattr(hparams, "num_residues", 512)  # safe default
        self.num_aa = getattr(hparams, "num_aa", 21)           # 20 + UNK

        self.edge_attr_dim = getattr(hparams, "edge_attr_dim", 0)

        # indices in x
        self.Z_idx = 0
        self.res_idx = 1
        self.aa_idx = 2
        self.xyz_start = 3
        self.xyz_end = 6

        # -------------------------
        # Node feat embed -> EGNN feat dim
        # -------------------------
        feat_in_dim = 3  # [Z,res,aa] as scalars
        act = nn.GELU()
        self.feat_embed = nn.Sequential(
            nn.Linear(feat_in_dim, 128),
            act,
            nn.Linear(128, self.input_dim),
        )

        # EGNN stack
        self.num_egnn_layers = getattr(hparams, "num_egnn_layers", getattr(hparams, "num_mp_layers", 3))
        self.egnn_layers = nn.ModuleList([
            EGNN(dim=self.input_dim, edge_dim=self.edge_attr_dim) for _ in range(self.num_egnn_layers)
        ])

        # scattering input: concat([feats, coors]) => input_dim + 3
        self.scatter_in_dim = self.input_dim + 3
        self.scattering_network = Scatter_layer(self.scatter_in_dim, self.max_seq_len, trainable_f=False)

        self.pos_encoder = PositionalEncoding(d_model=self.scattering_network.out_shape(), max_len=self.max_seq_len)

        self.row_encoder = TransformerEncoder(
            num_layers=self.layers,
            input_dim=self.scattering_network.out_shape(),
            num_heads=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=self.probs,
        )

        self.bottleneck_module = BaseBottleneck(self.scattering_network.out_shape(), self.hidden_dim, self.latent_dim)

        # -------------------------
        # Decoders
        # -------------------------
        # feature heads (per-node classification)
        self.Z_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_nodes * self.num_Z),
        )

        self.res_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_nodes * self.num_residues),
        )

        self.aa_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_nodes * self.num_aa),
        )

        # xyz head
        self.xyz_decoder = nn.Sequential(
            spectral_norm(nn.Linear(self.latent_dim, self.hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(self.hidden_dim, self.num_nodes * 3)),
        )

    def row_transformer_encoding(self, embedded_batch):
        pos_encoded = self.pos_encoder(embedded_batch)
        out = self.row_encoder(pos_encoded)
        att = self.row_encoder.get_attention_maps(pos_encoded)
        return out, att

    # ---- decoders ----
    def reconstruct_Z_logits(self, z_rep):
        h = self.Z_decoder(z_rep)
        return h.view(-1, self.num_nodes, self.num_Z)

    def reconstruct_res_logits(self, z_rep):
        h = self.res_decoder(z_rep)
        return h.view(-1, self.num_nodes, self.num_residues)

    def reconstruct_aa_logits(self, z_rep):
        h = self.aa_decoder(z_rep)
        return h.view(-1, self.num_nodes, self.num_aa)

    def reconstruct_xyz(self, z_rep):
        h = self.xyz_decoder(z_rep)
        return h.view(-1, self.num_nodes, 3)

    def encode(self, batch):
        x_gt_dense, node_mask = to_dense_batch(batch.x, batch.batch, max_num_nodes=self.num_nodes)
        x_in = batch.x.float()

        B = batch.num_graphs
        N = self.num_nodes

        # fixed-size sanity
        if x_in.size(0) != B * N:
            raise RuntimeError(f"Expected fixed-size graphs: got {x_in.size(0)} nodes, expected B*N={B*N}")

        # scalar features (Z,res,aa) as floats -> embed
        scalar_feat = x_in[:, :3]                 # (B*N,3)
        coors = x_in[:, 3:6]                      # (B*N,3)

        feat_dense = scalar_feat.view(B, N, 3)
        coor_dense = coors.view(B, N, 3)

        feats = self.feat_embed(feat_dense)       # (B,N,input_dim)
        coors = coor_dense                        # (B,N,3)

        # edges
        edges = None
        if hasattr(batch, "edge_attr") and batch.edge_attr is not None and self.edge_attr_dim > 0:
            edges = pyg_edges_to_dense_edges(
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch_vec=batch.batch,
                num_graphs=B,
                num_nodes=N,
                edge_dim=self.edge_attr_dim,
                device=feats.device,
            )

        # EGNN
        for layer in self.egnn_layers:
            if edges is None:
                feats, coors = layer(feats, coors)
            else:
                feats, coors = layer(feats, coors, edges)

        # scattering input
        feats_flat = feats.reshape(B * N, self.input_dim)
        coors_flat = coors.reshape(B * N, 3)

        x_orig = batch.x
        batch.x = torch.cat([feats_flat, coors_flat], dim=-1)  # (B*N,input_dim+3)

        coeffs = self.scattering_network(batch)
        if coeffs.ndim == 2:
            coeffs = coeffs.unsqueeze(0)

        batch.x = x_orig

        row_out, att_maps = self.row_transformer_encoding(coeffs)

        z_rep = row_out.sum(1)
        if z_rep.ndim == 1:
            z_rep = z_rep.unsqueeze(0)

        z_rep = self.bottleneck_module(z_rep)

        # decode
        Z_logits = self.reconstruct_Z_logits(z_rep)
        res_logits = self.reconstruct_res_logits(z_rep)
        aa_logits = self.reconstruct_aa_logits(z_rep)
        xyz_pred = self.reconstruct_xyz(z_rep)

        x_recon = (Z_logits, res_logits, aa_logits, xyz_pred)
        return z_rep, coeffs, att_maps, x_recon, x_gt_dense, node_mask

    def forward(self, batch):
        return self.encode(batch)

    # -------------------------
    # Reconstruction loss
    # -------------------------
    def recon_aa_loss(self, predictions, targets, mask=None):
        """
        predictions: (Z_logits, res_logits, aa_logits, xyz_pred)
        targets: x_gt_dense = (B,N,6) with [Z,res,aa,xyz]
        """
        Z_logits, res_logits, aa_logits, xyz_pred = predictions
        x_gt_dense = targets.float()

        if mask is None:
            mask = torch.ones(x_gt_dense.shape[:2], dtype=torch.bool, device=x_gt_dense.device)

        # integer targets (clamp into vocab ranges)
        Z_tgt = x_gt_dense[..., 0].round().long().clamp(0, self.num_Z - 1)
        res_tgt = x_gt_dense[..., 1].round().long().clamp(0, self.num_residues - 1)
        aa_tgt = x_gt_dense[..., 2].round().long().clamp(0, self.num_aa - 1)

        # flatten with mask
        mask_flat = mask.reshape(-1)

        def masked_ce(logits, tgt):
            logits_f = logits.reshape(-1, logits.size(-1))
            tgt_f = tgt.reshape(-1)
            if mask_flat.any():
                ce_all = F.cross_entropy(logits_f, tgt_f, reduction="none")
                return ce_all[mask_flat].mean()
            else:
                return torch.zeros((), device=logits.device)

        z_ce = masked_ce(Z_logits, Z_tgt)
        res_ce = masked_ce(res_logits, res_tgt)
        aa_ce = masked_ce(aa_logits, aa_tgt)

        # xyz Kabsch
        xyz_gt = x_gt_dense[..., 3:6]
        coord_loss = kabsch_mse_loss(
            pred_xyz=xyz_pred,
            true_xyz=xyz_gt,
            mask=mask,
            allow_reflection=self.allow_reflection,
        )

        # total node loss (you can weight these if you want)
        feat_ce = z_ce + res_ce + aa_ce
        node_total = feat_ce + self.coord_weight * coord_loss

        return _scalarize(node_total), _scalarize(feat_ce), _scalarize(coord_loss)
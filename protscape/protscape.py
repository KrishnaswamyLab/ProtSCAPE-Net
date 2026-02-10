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
from protscape.gcn_layers import GCN_Layer, SimpleGCN_Layer


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
        # ABLATION CONFIGURATIONS
        # -------------------------
        # Feature extractor type: "scattering", "gcn", "simple_gcn"
        self.feature_extractor = getattr(hparams, "feature_extractor", "scattering")
        self.gcn_num_layers = getattr(hparams, "gcn_num_layers", 4)
        self.gcn_hidden_channels = getattr(hparams, "gcn_hidden_channels", None)
        
        # Node feature ablations
        node_ablate = getattr(hparams, "ablate_node_features", {})
        self.use_atomic_number = node_ablate.get("use_atomic_number", True)
        self.use_residue_index = node_ablate.get("use_residue_index", True)
        self.use_amino_acid = node_ablate.get("use_amino_acid", True)
        self.use_xyz = node_ablate.get("use_xyz", True)
        self.randomize_atomic_number = node_ablate.get("randomize_atomic_number", False)
        self.randomize_residue_index = node_ablate.get("randomize_residue_index", False)
        self.randomize_amino_acid = node_ablate.get("randomize_amino_acid", False)
        self.randomize_xyz = node_ablate.get("randomize_xyz", False)
        
        # Edge feature ablations
        edge_ablate = getattr(hparams, "ablate_edge_features", {})
        self.use_edge_features = edge_ablate.get("use_edge_features", True)
        self.randomize_edge_features = edge_ablate.get("randomize_edge_features", False)
        self.zero_edge_features = edge_ablate.get("zero_edge_features", False)
        
        # Pre-compute feature indices for efficient slicing (optimization)
        self._feature_indices = []
        if self.use_atomic_number:
            self._feature_indices.append(0)
        if self.use_residue_index:
            self._feature_indices.append(1)
        if self.use_amino_acid:
            self._feature_indices.append(2)
        # Convert to tensor for efficient indexing
        self._feature_idx_tensor = torch.tensor(self._feature_indices, dtype=torch.long)

        # -------------------------
        # Node feat embed -> EGNN feat dim
        # -------------------------
        # Calculate actual feature dimension based on ablations
        self.active_node_features = sum([
            self.use_atomic_number,
            self.use_residue_index,
            self.use_amino_acid
        ])
        
        if self.active_node_features == 0:
            raise ValueError("At least one node feature (Z, res, or aa) must be enabled")
        
        feat_in_dim = self.active_node_features
        act = nn.GELU()
        self.feat_embed = nn.Sequential(
            nn.Linear(feat_in_dim, 128),
            act,
            nn.Linear(128, self.input_dim),
        )

        # EGNN stack - adjust edge_dim based on edge ablation
        self.num_egnn_layers = getattr(hparams, "num_egnn_layers", getattr(hparams, "num_mp_layers", 3))
        effective_edge_dim = 0 if (not self.use_edge_features or self.zero_edge_features) else self.edge_attr_dim
        self.egnn_layers = nn.ModuleList([
            EGNN(dim=self.input_dim, edge_dim=effective_edge_dim) for _ in range(self.num_egnn_layers)
        ])

        # Feature extraction layer: scattering or GCN variants
        # Adjust input dimension based on xyz ablation
        xyz_dim = 3 if self.use_xyz else 0
        self.scatter_in_dim = self.input_dim + xyz_dim
        
        if self.feature_extractor == "scattering":
            self.feature_extraction_network = Scatter_layer(self.scatter_in_dim, self.max_seq_len, trainable_f=False)
        elif self.feature_extractor == "gcn":
            self.feature_extraction_network = GCN_Layer(
                self.scatter_in_dim, 
                self.max_seq_len, 
                num_layers=self.gcn_num_layers,
                hidden_channels=self.gcn_hidden_channels
            )
        elif self.feature_extractor == "simple_gcn":
            self.feature_extraction_network = SimpleGCN_Layer(
                self.scatter_in_dim,
                self.max_seq_len,
                num_layers=self.gcn_num_layers,
                output_dim=self.scatter_in_dim * self.gcn_num_layers
            )
        else:
            raise ValueError(f"Unknown feature_extractor: {self.feature_extractor}. Use 'scattering', 'gcn', or 'simple_gcn'.")

        self.pos_encoder = PositionalEncoding(d_model=self.feature_extraction_network.out_shape(), max_len=self.max_seq_len)

        self.row_encoder = TransformerEncoder(
            num_layers=self.layers,
            input_dim=self.feature_extraction_network.out_shape(),
            num_heads=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=self.probs,
        )

        self.bottleneck_module = BaseBottleneck(self.feature_extraction_network.out_shape(), self.hidden_dim, self.latent_dim)

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
    
    def apply_node_feature_ablations(self, feat_dense, coor_dense):
        """
        Apply node feature ablations by actually removing features from the input.
        OPTIMIZED: Uses pre-computed indices for efficient slicing.
        
        Args:
            feat_dense: (B, N, 3) - [Z, res, aa]
            coor_dense: (B, N, 3) - xyz coordinates
            
        Returns:
            Modified feat_dense with only active features, coor_dense (or None if ablated)
        """
        B, N, _ = feat_dense.shape
        device = feat_dense.device
        
        # OPTIMIZATION: Use direct indexing instead of list building + concatenation
        # This is MUCH faster than building a list and concatenating
        if self.randomize_atomic_number or self.randomize_residue_index or self.randomize_amino_acid:
            # Slow path: need to handle randomization
            active_features = []
            
            if self.use_atomic_number:
                if self.randomize_atomic_number:
                    feat = torch.randint(0, self.num_Z, (B, N, 1), device=device, dtype=feat_dense.dtype)
                else:
                    feat = feat_dense[:, :, 0:1]
                active_features.append(feat)
            
            if self.use_residue_index:
                if self.randomize_residue_index:
                    feat = torch.randint(0, self.num_residues, (B, N, 1), device=device, dtype=feat_dense.dtype)
                else:
                    feat = feat_dense[:, :, 1:2]
                active_features.append(feat)
            
            if self.use_amino_acid:
                if self.randomize_amino_acid:
                    feat = torch.randint(0, self.num_aa, (B, N, 1), device=device, dtype=feat_dense.dtype)
                else:
                    feat = feat_dense[:, :, 2:3]
                active_features.append(feat)
            
            feat_dense = torch.cat(active_features, dim=-1)
        else:
            # FAST path: direct indexing with pre-computed indices (no list/cat overhead)
            idx_tensor = self._feature_idx_tensor.to(device)
            feat_dense = feat_dense.index_select(dim=2, index=idx_tensor)
        
        # Handle coordinates
        if not self.use_xyz:
            coor_dense = None  # Completely remove xyz
        elif self.randomize_xyz:
            # Random coordinates in reasonable range (e.g., -5 to 5 nm)
            coor_dense = torch.randn_like(coor_dense) * 2.0
        
        return feat_dense, coor_dense
    
    def apply_edge_feature_ablations(self, edges, B, N):
        """
        Apply edge feature ablations by completely removing them.
        
        Args:
            edges: (B, N, N, edge_dim) dense edge features or None
            B: batch size
            N: number of nodes
            
        Returns:
            Modified edges or None (None means no edge features)
        """
        if edges is None:
            return None
        
        # If ablating edge features, return None (complete removal)
        if not self.use_edge_features or self.zero_edge_features:
            return None
        
        # If randomizing, replace with random features (not ablation, but useful for control)
        if self.randomize_edge_features:
            edges = torch.randn_like(edges)
        
        return edges

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
        
        # Apply node feature ablations
        feat_dense, coor_dense = self.apply_node_feature_ablations(feat_dense, coor_dense)

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
            # Apply edge feature ablations
            edges = self.apply_edge_feature_ablations(edges, B, N)

        # EGNN
        for layer in self.egnn_layers:
            if edges is None:
                feats, coors = layer(feats, coors)
            else:
                feats, coors = layer(feats, coors, edges)

        # scattering input
        feats_flat = feats.reshape(B * N, self.input_dim)
        
        x_orig = batch.x
        
        # Concatenate coordinates only if not ablated
        if coors is not None:
            coors_flat = coors.reshape(B * N, 3)
            batch.x = torch.cat([feats_flat, coors_flat], dim=-1)  # (B*N, input_dim+3)
        else:
            batch.x = feats_flat  # (B*N, input_dim) - no coordinates

        coeffs = self.feature_extraction_network(batch)
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
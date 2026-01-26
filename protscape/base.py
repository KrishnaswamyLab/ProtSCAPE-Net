"""Base model for TGTransformer with ATLAS-specific losses.
Implements:
- KL divergence in latent space (OPTIONAL)
- Energy Laplacian smoothness loss
- training_step and validation_step with multi_loss
- multi_loss combining Laplacian smoothness and AA reconstruction loss"""

import argparse
import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from protscape.scheduler import CosineWarmupScheduler
import torch
import torch.nn.functional as F

def kl_divergence(mu, logvar):
    """
    Compute KL divergence between N(mu, var) and N(0, 1)
    mu: (B, D)
    logvar: (B, D)
    returns: scalar
    """
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
    return kld.mean()

def energy_laplacian_smoothness(
    z: torch.Tensor,        # (B,D)
    y: torch.Tensor,        # (B,)
    sigma2: float = 1.0,    # bandwidth in latent
    eps: float = 1e-8,
    detach_kernel: bool = True,
):
    """
    L = sum_{i!=j} w_ij * (y_i - y_j)^2 / sum w_ij
    where w_ij = exp(-||z_i-z_j||^2 / (2*sigma2))

    This is exactly the graph smoothness of y over a latent-space affinity graph.
    """
    z_sqdist = torch.cdist(z, z).pow(2)   # (B, B)

    y_diff = y[:, None] - y[None, :]
    y_sqdist = (y_diff ** 2).sum(dim=-1)

    return 0.5 * (torch.exp(-0.5 * z_sqdist * 1/(2*sigma2)) * y_sqdist).mean()


class TGTransformerBaseModel_ATLAS(LightningModule):
    def __init__(self, hparams=None):
        super(TGTransformerBaseModel_ATLAS, self).__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()
        self.energy_axis = torch.nn.Parameter(torch.randn(128))  # for rank loss
        self.lr = hparams.lr
        self.task = getattr(hparams, "task", "reg")  # unused now
        self.batch_size = getattr(hparams, "batch_size", 1)

        self.alpha = getattr(hparams, "alpha", 1.0)       # weight on Laplacian smoothness
        self.beta_loss = getattr(hparams, "beta_loss", 1.0)

        # Laplacian graph hyperparams
        self.lap_sigma_mode = getattr(hparams, "lap_sigma_mode", "median")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_sched = CosineWarmupScheduler(opt, warmup=5000, max_iters=50000)
        return [opt], [lr_sched]

    def shared_step(self, batch):
        """
        Supports:
        A) Old: (z_rep, coeffs, att_maps, x_recon, x_gt, node_mask)           len=6
        B) New ablation: (z_rep, x_recon, x_gt, node_mask)                    len=4
        C) Older w/ energy head: (preds, z_rep, coeffs, att_maps, x_recon, x_gt, node_mask) len=7
        """
        e = batch.energy
        out = self(batch)

        if isinstance(out, (tuple, list)):
            if len(out) == 4:
                z_rep, x_recon, x_gt, node_mask = out
                return z_rep, e, x_recon, x_gt, node_mask

            if len(out) == 6:
                z_rep, coeffs, att_maps, x_recon, x_gt, node_mask = out
                return z_rep, e, x_recon, x_gt, node_mask

            if len(out) == 7:
                preds, z_rep, coeffs, att_maps, x_recon, x_gt, node_mask = out
                return z_rep, e, x_recon, x_gt, node_mask

        raise ValueError(
            f"Unexpected forward() output format: {type(out)} "
            f"with len={len(out) if isinstance(out,(tuple,list)) else 'NA'}"
        )


    def relabel(self, loss_dict, label):
        return {label + str(key): val for key, val in loss_dict.items()}

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        z_rep, energies, x_recon, x_gt, node_mask = self.shared_step(batch)

        train_loss, train_logs = self.multi_loss(
            z_rep=z_rep,
            energies=energies,
            aa_recon=x_recon,
            aa_gt=x_gt,
            node_mask=node_mask,
        )
        train_logs = self.relabel(train_logs, "train_")
        self.log_dict(train_logs, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return train_loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        z_rep,  energies, x_recon, x_gt, node_mask = self.shared_step(batch)

        val_loss, val_logs = self.multi_loss(
            z_rep=z_rep,
            energies=energies,
            aa_recon=x_recon,
            aa_gt=x_gt,
            node_mask=node_mask,
        )
        val_logs = self.relabel(val_logs, "val_")
        self.log_dict(val_logs, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return val_loss

    def multi_loss(self, *args, **kwargs):
        """
        total = alpha * laplacian_smoothness(z_rep, energy) + beta_loss * node_total

        node_total comes from child recon_aa_loss():
          node_total = type_ce + coord_weight * coord_loss
        """
        z_rep = kwargs["z_rep"]
        energies = kwargs["energies"]
        aa_recon = kwargs["aa_recon"]
        aa_gt = kwargs["aa_gt"]
        node_mask = kwargs.get("node_mask", None)

        # Laplacian smoothness replaces energy-prediction loss
        device = z_rep.device
        e = torch.as_tensor(energies, dtype=torch.float32, device=device).view(-1)
        # lap_loss = lap_smooth_soft(Z=z_rep, e=e, sigma2=1.0)
        lap_loss = energy_laplacian_smoothness(z_rep, e, sigma2=6.0, detach_kernel=False)
         # AA reconstruction loss
        node_total, type_ce, coord_loss = self.recon_aa_loss(
            predictions=aa_recon, targets=aa_gt, mask=node_mask
        )

        alpha = getattr(self, "alpha", 1.0)
        beta = getattr(self, "beta_loss", 1.0)

        total_loss = alpha * lap_loss + beta * node_total

        log_losses = {
            "total_loss": total_loss.detach(),
            "lap_smooth_loss": (lap_loss).detach(),
            "node_loss": (node_total).detach(),
            "type_ce_loss": (type_ce).detach(),
            "coord_loss": (coord_loss).detach(),
        }
        return total_loss, log_losses

    def recon_aa_loss(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError
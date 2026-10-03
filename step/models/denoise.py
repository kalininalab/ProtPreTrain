import math
from typing import Any, Literal

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torchmetrics as metrics
import wandb
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from torchmetrics import ConfusionMatrix

from ..data.parsers import THREE_TO_ONE
from ..utils import WarmUpCosineLR, plot_aa_tsne, plot_confmat, plot_node_embeddings
from .downstream import SimpleMLP


class RedrawProjection:
    """Used for performer stuff"""

    def __init__(self, model: torch.nn.Module, redraw_interval: int = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        """Recalculates projections."""
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules() if isinstance(module, pyg.nn.attention.PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


def sequence_pe(batch: Data, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of each residue's index within its chain (nodes are stored in sequence order)."""
    idx = torch.arange(batch.num_nodes, device=batch.x.device)
    if batch.batch is not None:
        idx = idx - batch.ptr[batch.batch]
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2, device=idx.device) / dim)
    angles = idx.unsqueeze(-1).float() * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class DistanceRBF(torch.nn.Module):
    """Expand edge lengths into Gaussian radial basis features on [0, cutoff]."""

    def __init__(self, num_rbf: int, cutoff: float):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, cutoff, num_rbf))
        self.gamma = (num_rbf / cutoff) ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Map (E,) distances to (E, num_rbf) features."""
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)


class EquivariantNoiseHead(torch.nn.Module):
    """Predict per-node noise as a learned weighting of unit vectors to neighbours.

    Weights depend only on invariant features, so rotating the input structure rotates the prediction with it.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.weight = SimpleMLP(2 * hidden_dim + edge_dim, hidden_dim, 1, dropout)

    def forward(self, x, pos, edge_index, edge_attr) -> torch.Tensor:
        """Return (N, 3) noise predictions."""
        src, dst = edge_index
        vec = pos[dst] - pos[src]
        vec = vec / vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        w = self.weight(torch.cat([x[dst], x[src], edge_attr], dim=-1))
        return scatter(w * vec, dst, dim=0, dim_size=x.size(0), reduce="mean")


class DenoiseModel(LightningModule):
    """Uses GraphGPS transformer layers to encode the graph and predict noise and node type.

    Constructor defaults reproduce the original architecture, so older checkpoints load unchanged;
    train.py sets the improved defaults.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        pe_dim: int = 64,
        pos_dim: int = 64,
        num_layers: int = 6,
        heads: int = 8,
        attn_type: Literal["multihead", "performer"] = "performer",
        dropout: float = 0.5,
        alpha: float = 0.5,
        predict_all: bool = False,
        pe: Literal["rw", "seq", "none"] = "rw",
        walk_length: int = 20,
        edge_dim: int = 0,
        radius: float = 10,
        invariant: bool = False,
        lr: float = 1e-4,
        scheduler: Literal["cosine", "legacy"] = "legacy",
        warmup_frac: float = 0.05,
        **kwargs,
    ):
        super(DenoiseModel, self).__init__()
        if pe == "none":
            pe_dim = 0
        if invariant:
            # Raw coordinates are frame-dependent, so the invariant model sees geometry only through edge lengths
            assert edge_dim > 0, "invariant model needs edge distance features (edge_dim > 0)"
            pos_dim = 0
        assert pe_dim % 2 == 0, "pe_dim must be even"
        assert hidden_dim > (pos_dim + pe_dim)
        self.save_hyperparameters()
        self.lr = lr
        self.alpha = alpha
        self.predict_all = predict_all
        self.pe = pe
        self.pe_dim = pe_dim
        self.invariant = invariant
        self.feat_encode = torch.nn.Embedding(21, hidden_dim - pos_dim - pe_dim)
        if pos_dim > 0:
            self.pos_encode = torch.nn.Linear(3, pos_dim)
        if pe == "rw":
            self.pe_norm = torch.nn.BatchNorm1d(walk_length)
            self.pe_encode = torch.nn.Linear(walk_length, pe_dim)
        if edge_dim > 0:
            self.rbf = DistanceRBF(edge_dim, radius)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            local = pyg.nn.GINEConv(nn, edge_dim=edge_dim) if edge_dim > 0 else pyg.nn.GINConv(nn)
            conv = pyg.nn.GPSConv(
                hidden_dim, local, heads=heads, attn_type=attn_type, attn_kwargs={"dropout": dropout}
            )
            self.convs.append(conv)
        if invariant:
            self.noise_pred = EquivariantNoiseHead(hidden_dim, edge_dim, dropout)
        else:
            self.noise_pred = SimpleMLP(hidden_dim, hidden_dim, 3, dropout)
        self.type_pred = SimpleMLP(hidden_dim, hidden_dim, 20, dropout)
        self.aggr = pyg.nn.aggr.MeanAggregation()
        self.redraw_projection = RedrawProjection(
            self.convs, redraw_interval=1000 if attn_type == "performer" else None
        )

    def edge_features(self, batch: Data) -> torch.Tensor | None:
        """RBF-expanded edge lengths, or None when the model uses no edge features."""
        if self.hparams.edge_dim == 0:
            return None
        src, dst = batch.edge_index
        return self.rbf((batch.pos[dst] - batch.pos[src]).norm(dim=-1))

    def encode(self, batch: Data, edge_attr: torch.Tensor | None) -> torch.Tensor:
        """Per-node embeddings from residue types, positional encoding, and (unless invariant) raw coordinates."""
        feats = [self.feat_encode(batch.x)]
        if not self.invariant:
            feats.append(self.pos_encode(batch.pos))
        if self.pe == "rw":
            feats.append(self.pe_encode(self.pe_norm(batch.pe)))
        elif self.pe == "seq":
            feats.append(sequence_pe(batch, self.pe_dim))
        x = torch.cat(feats, dim=1)
        for conv in self.convs:
            if edge_attr is None:
                x = conv(x, batch.edge_index, batch.batch)
            else:
                x = conv(x, batch.edge_index, batch.batch, edge_attr=edge_attr)
        return x

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        self.redraw_projection.redraw_projections()
        edge_attr = self.edge_features(batch)
        x = self.encode(batch, edge_attr)
        if self.predict_all:
            batch.type_pred = self.type_pred(x)
        else:
            batch.type_pred = self.type_pred(x[batch.mask])
        if self.invariant:
            batch.noise_pred = self.noise_pred(x, batch.pos, batch.edge_index, edge_attr)
        else:
            batch.noise_pred = self.noise_pred(x)
        batch.x = x
        return batch

    def log_confmat(self):
        """Log confusion matrix to wandb."""
        confmat_df = self.confmat.compute().detach().cpu().numpy()
        indices = list(THREE_TO_ONE)[:-1]
        confmat_df = pd.DataFrame(confmat_df, index=indices, columns=indices).round(2)
        self.confmat.reset()
        return plot_confmat(confmat_df)

    def log_aa_embed(self):
        """Log t-SNE plot of amino acid embeddings."""
        aa = torch.tensor(range(21), dtype=torch.long, device=self.device)
        emb = self.feat_encode(aa).detach().cpu()
        return plot_aa_tsne(emb)

    def log_figs(self, step: str):
        """Log figures to wandb."""
        test_batch = self.forward(self.test_batch.clone())
        node_pca = plot_node_embeddings(
            test_batch.x, self.test_batch.x, [self.test_batch.uniprot_id[x] for x in self.test_batch.batch]
        )
        figs = {
            f"{step}/confmat": self.log_confmat(),
            f"{step}/aa_pca": self.log_aa_embed(),
            f"{step}/node_pca": node_pca,
        }
        wandb.log(figs)

    def _shared_step(self, batch: Data, stage: str) -> torch.Tensor:
        batch = self.forward(batch)
        noise_loss = F.mse_loss(batch.noise_pred, batch.noise)
        target = batch.orig_x if self.predict_all else batch.orig_x[batch.mask]
        pred_loss = F.cross_entropy(batch.type_pred, target)
        acc = metrics.functional.accuracy(batch.type_pred, target, task="multiclass", num_classes=20)
        loss = noise_loss * self.alpha + (1 - self.alpha) * pred_loss
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/noise_loss": noise_loss,
                f"{stage}/pred_loss": pred_loss,
                f"{stage}/pred_acc": acc,
            },
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: Data, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Denoising + masked type prediction loss."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Data, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Same losses on held-out structures."""
        return self._shared_step(batch, "val")

    def predict_step(self, batch: Any, batch_idx: int) -> Data:
        """Return updated batch with all the information."""
        x = self.encode(batch, self.edge_features(batch))
        batch.aggr_x = self.aggr(x, batch.batch)
        return batch

    def configure_optimizers(self) -> Any:
        """interval is making sure you step after each step, not each epoch"""
        optim = torch.optim.AdamW(self.parameters(), self.lr)
        if self.hparams.scheduler == "legacy":
            scheduler = WarmUpCosineLR(
                optim, warmup_steps=10000, start_lr=1e-5, max_lr=self.lr, min_lr=1e-7, cycle_len=100000
            )
        else:
            # One warmup + cosine decay spanning exactly the run, whatever its length
            total = int(self.trainer.estimated_stepping_batches)
            warmup = max(1, int(self.hparams.warmup_frac * total))
            scheduler = WarmUpCosineLR(
                optim,
                warmup_steps=warmup,
                start_lr=self.lr / 100,
                max_lr=self.lr,
                min_lr=self.lr / 100,
                cycle_len=max(1, total - warmup),
            )
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

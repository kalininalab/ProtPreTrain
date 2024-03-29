from typing import Any, Literal

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torchmetrics as metrics
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from torchmetrics import ConfusionMatrix

import wandb

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


class DenoiseModel(LightningModule):
    """Uses GraphGPS transformer layers to encode the graph and predict noise and node type."""

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
        predict_all: bool = True,
        walk_length: int = 20,
        lr: float = 1e-4,
        **kwargs,
    ):
        super(DenoiseModel, self).__init__()
        assert hidden_dim > (pos_dim + pe_dim)
        self.save_hyperparameters()
        self.lr = lr
        self.alpha = alpha
        self.predict_all = predict_all
        self.feat_encode = torch.nn.Embedding(21, hidden_dim - pos_dim - pe_dim)
        self.pos_encode = torch.nn.Linear(3, pos_dim)
        self.pe_norm = torch.nn.BatchNorm1d(walk_length)
        self.pe_encode = torch.nn.Linear(walk_length, pe_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            conv = pyg.nn.GPSConv(
                hidden_dim, pyg.nn.GINConv(nn), heads=heads, attn_type=attn_type, attn_kwargs={"dropout": dropout}
            )
            self.convs.append(conv)
        self.noise_pred = SimpleMLP(hidden_dim, hidden_dim, 3, dropout)
        self.type_pred = SimpleMLP(hidden_dim, hidden_dim, 20, dropout)
        self.aggr = pyg.nn.aggr.MeanAggregation()
        self.redraw_projection = RedrawProjection(
            self.convs, redraw_interval=1000 if attn_type == "performer" else None
        )

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        self.redraw_projection.redraw_projections()
        x = self.feat_encode(batch.x)
        pos = self.pos_encode(batch.pos)
        pe = self.pe_norm(batch.pe)
        pe = self.pe_encode(pe)
        x = torch.cat([x, pos, pe], dim=1)
        for conv in self.convs:
            x = conv(x, batch.edge_index, batch.batch)
        if self.predict_all:
            batch.type_pred = self.type_pred(x)
        else:
            batch.type_pred = self.type_pred(x[batch.mask])
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

    def training_step(self, batch: Data, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """Shared step for training and validation."""
        batch = self.forward(batch)
        noise_loss = F.mse_loss(batch.noise_pred, batch.noise)
        if self.predict_all:
            pred_loss = F.cross_entropy(batch.type_pred, batch.orig_x)
            acc = metrics.functional.accuracy(batch.type_pred, batch.orig_x, task="multiclass", num_classes=20)
        else:
            pred_loss = F.cross_entropy(batch.type_pred, batch.orig_x[batch.mask])
            acc = metrics.functional.accuracy(
                batch.type_pred, batch.orig_x[batch.mask], task="multiclass", num_classes=20
            )
        loss = noise_loss * self.alpha + (1 - self.alpha) * pred_loss
        self.log_dict(
            {"train/loss": loss, "train/noise_loss": noise_loss, "train/pred_loss": pred_loss, "train/pred_acc": acc},
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch: Any, batch_idx: int) -> Data:
        """Return updated batch with all the information."""
        x = self.feat_encode(batch.x)
        pos = self.pos_encode(batch.pos)
        pe = self.pe_norm(batch.pe)
        pe = self.pe_encode(batch.pe)
        x = torch.cat([x, pos, pe], dim=1)
        for conv in self.convs:
            x = conv(x, batch.edge_index, batch.batch)
        batch.aggr_x = self.aggr(x, batch.batch)
        return batch

    def configure_optimizers(self) -> Any:
        """interval is making sure you step after each step, not each epoch"""
        optim = torch.optim.AdamW(self.parameters(), self.lr)
        scheduler = WarmUpCosineLR(
            optim, warmup_steps=10000, start_lr=1e-5, max_lr=self.lr, min_lr=1e-7, cycle_len=100000
        )
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

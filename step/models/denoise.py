from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from graphgps.layer.gps_layer import GPSLayer
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from torchmetrics import ConfusionMatrix

import wandb

from ..data.parsers import THREE_TO_ONE
from ..utils import plot_aa_tsne, plot_confmat, plot_node_embeddings
from .downstream import LazySimpleMLP, SimpleMLP


class DenoiseModel(LightningModule):
    """Uses GraphGPS transformer layers to encode the graph and predict noise and node type."""

    def __init__(
        self,
        local_module: str = "GAT",
        global_module: str = "Performer",
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        alpha: float = 1.0,
        weighted_loss: bool = False,
        predict_all: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.weighted_loss = weighted_loss
        self.alpha = alpha
        self.predict_all = predict_all
        self.feat_encode = torch.nn.Embedding(21, hidden_dim)
        self.edge_encode = torch.nn.Linear(3, hidden_dim)
        self.node_encode = torch.nn.Sequential(
            *[
                GPSLayer(
                    hidden_dim,
                    local_module,
                    global_module,
                    num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.noise_pred = SimpleMLP(hidden_dim, hidden_dim, 3, dropout)
        self.type_pred = SimpleMLP(hidden_dim, hidden_dim, 20, dropout)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=20, normalize="true")
        self.aggr = torch_geometric.nn.aggr.MeanAggregation()

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        batch.x = self.feat_encode(batch.x)
        batch.edge_attr = self.edge_encode(batch.edge_attr)
        batch = self.node_encode(batch)
        if self.predict_all:
            batch.type_pred = self.type_pred(batch.x)
        else:
            batch.type_pred = self.type_pred(batch.x[batch.mask])
        batch.noise_pred = self.noise_pred(batch.x)
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

    def training_step(self, batch: Data, idx) -> dict:
        """Shared step for training and validation."""
        sch = self.lr_schedulers()
        sch.step()
        if self.global_step == 0:
            self.test_batch = batch.clone()
        batch = self.forward(batch)
        noise_loss = F.mse_loss(batch.noise_pred, batch.noise)
        if self.predict_all:
            pred_loss = F.cross_entropy(batch.type_pred, batch.orig_x)
            self.confmat.update(batch.type_pred, batch.orig_x)
        else:
            pred_loss = F.cross_entropy(batch.type_pred, batch.orig_x[batch.mask])
            self.confmat.update(batch.type_pred, batch.orig_x[batch.mask])
        loss = noise_loss + self.alpha * pred_loss
        # acc = accuracy(batch.type_pred, batch.orig_x, task="multiclass")
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=batch.num_graphs)
        if self.global_step % 100 == 0:
            wandb.log(
                {
                    "train/loss": loss,
                    # f"{step}/acc": acc,
                    "train/noise_loss": noise_loss,
                    "train/pred_loss": pred_loss,
                }
            )
        if self.global_step % 1000 == 0:
            self.log_figs("train")
        return dict(
            loss=loss,
            noise_loss=noise_loss.detach(),
            pred_loss=pred_loss.detach(),
            # acc=acc.detach(),
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Data:
        """Return updated batch with all the information."""
        batch.x = self.feat_encode(batch.x)
        batch.edge_attr = self.edge_encode(batch.edge_attr)
        batch = self.node_encode(batch)
        batch.aggr_x = self.aggr(batch.x, batch.batch)
        return batch

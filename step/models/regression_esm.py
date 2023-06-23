import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics
from pytorch_lightning import LightningModule
from torch_geometric.data import Data

import wandb

from .utils import SimpleMLP


class RegressionESMModel(LightningModule):
    """Uses ESM."""

    def __init__(
        self,
        in_dim: int = 1280,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.linear = SimpleMLP(in_dim, hidden_dim, 1, dropout)

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        return self.linear(batch.x.view(batch.num_graphs, -1))

    def shared_step(self, batch: Data, step_name: str) -> dict:
        """Shared step for training and validation."""
        y = self.forward(batch)
        pred_y = batch.y.unsqueeze(-1)
        loss = F.mse_loss(pred_y, y)
        mae = metrics.mean_absolute_error(pred_y, y)
        r2 = metrics.r2_score(pred_y, y)
        self.log(f"{step_name}/loss", loss, batch_size=batch.num_graphs)
        self.log(f"{step_name}/mae", mae, batch_size=batch.num_graphs)
        self.log(f"{step_name}/r2", r2, batch_size=batch.num_graphs)
        return dict(
            loss=loss,
            mae=mae,
            r2=r2,
        )

    def training_step(self, batch: Data, batch_idx: int) -> dict:
        """Training step."""
        if self.global_step == 0:
            wandb.define_metric("val/loss", summary="min")
        return self.shared_step(batch, "train")

    def validation_step(self, batch: Data, batch_idx: int) -> dict:
        """Validation step."""
        return self.shared_step(batch, "val")

    def test_step(self, batch: Data, batch_idx: int) -> dict:
        """Test step."""
        return self.shared_step(batch, "test")

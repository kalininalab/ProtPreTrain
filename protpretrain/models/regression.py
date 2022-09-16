import torch
import torch.nn.functional as F
import wandb
from torch_geometric.data import Data
from torchmetrics.functional.regression import explained_variance, mean_absolute_error, r2_score

from .classification import ClassificationModel


class RegressionModel(ClassificationModel):
    """Uses pretrained denoising model for a regression task."""

    def __init__(self, pretrained_checkpoint: str):
        super().__init__(pretrained_checkpoint, num_classes=1)

    def forward(self, batch: Data) -> Data:
        """Return batch prediction"""
        graph_embeddings = self.encoder.encode(batch)
        pred = self.mlp(graph_embeddings)
        return pred

    def shared_step(self, batch: Data, step: str) -> dict:
        """Shared training and validation step"""
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)
        mae = mean_absolute_error(pred, batch.y)
        r2 = r2_score(pred, batch.y)
        ev = explained_variance(pred, batch.y)
        self.log(f"{step}/loss", loss)
        self.log(f"{step}/mae", mae)
        self.log(f"{step}/r2", r2)
        self.log(f"{step}/ev", ev)
        return loss

    def training_step(self, batch):
        return self.shared_step(batch, "train")

    def validation_step(self, batch):
        return self.shared_step(batch, "val")

    def test_step(self, batch):
        return self.shared_step(batch, "test")

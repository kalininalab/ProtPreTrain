import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from torchmetrics.functional.classification import accuracy, auroc, matthews_corrcoef

from .denoise import DenoiseModel


class ClassificationModel(LightningModule):
    """Uses pretrained denoising model for a classification task."""

    def __init__(self, pretrained_checkpoint: str, num_classes: int):
        super().__init__()
        self.encoder = DenoiseModel.load_from_checkpoint(pretrained_checkpoint)
        self.encoder.freeze()
        hidden_dim = self.encoder.hidden_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: Data) -> Data:
        """Return batch prediction"""
        graph_embeddings = self.encoder.encode(batch)
        pred = self.mlp(graph_embeddings)
        return pred

    def shared_step(self, batch: Data, step: str) -> dict:
        """Shared training and validation step"""
        pred = self(batch)
        loss = F.cross_entropy(pred, batch.y)
        acc = accuracy(pred, batch.y)
        auc = auroc(pred, batch.y)
        mcc = matthews_corrcoef(pred, batch.y)
        self.log(f"{step}/loss", loss)
        self.log(f"{step}/accuracy", acc)
        self.log(f"{step}/auroc", auc)
        self.log(f"{step}/mcc", mcc)
        return loss

    def training_step(self, batch):
        return self.shared_step(batch, "train")

    def validation_step(self, batch):
        return self.shared_step(batch, "val")

    def test_step(self, batch):
        return self.shared_step(batch, "test")

from typing import Any

import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics
from pytorch_lightning import LightningModule
from torch_geometric.data import Data

import wandb


class LazySimpleMLP(torch.nn.Module):
    """Simple MLP for output heads, guesses the input dim on first pass."""

    def __init__(self, hid_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.LazyLinear(hid_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        """"""
        return self.main(x)


class SimpleMLP(torch.nn.Module):
    """Simple MLP for output heads."""

    def __init__(self, inp_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(inp_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        """"""
        return self.main(x)


class BaseModel(LightningModule):
    """Base class for all downstream stuff."""

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

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

    def configure_optimizers(self) -> Any:
        """Configure optimizers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=20,
            min_lr=1e-7,
            verbose=True,
        )
        return [optimizer], [{"scheduler": scheduler, "monitor": "val/loss"}]


class RegressionModel(BaseModel):
    """Base for regression, only defines shared steps"""

    def __init__(
        self,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.linear = LazySimpleMLP(hidden_dim, 1, dropout)

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        return self.linear(batch.x.view(batch.num_graphs, -1))

    def shared_step(self, batch: Data, step_name: str) -> dict:
        """Shared step for training and validation."""
        y_hat = self.forward(batch).squeeze(-1)
        y = batch.y
        loss = F.huber_loss(y_hat, y, delta=0.5)
        mae = metrics.mean_absolute_error(y_hat, y)
        r2 = metrics.r2_score(y_hat, y)
        corrcoef = metrics.spearman_corrcoef(y_hat, y)
        self.log(f"{step_name}/loss", loss, batch_size=batch.num_graphs)
        self.log(f"{step_name}/mae", mae, batch_size=batch.num_graphs)
        self.log(f"{step_name}/r2", r2, batch_size=batch.num_graphs)
        self.log(f"{step_name}/corrcoef", corrcoef, batch_size=batch.num_graphs)
        return dict(
            loss=loss,
            mae=mae,
            r2=r2,
            corrcoef=corrcoef,
        )


class ClassificationModel(BaseModel):
    """Takes in precomputed embeddings and predicts a class"""

    def __init__(
        self,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.linear = LazySimpleMLP(hidden_dim, 1, dropout)

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        return self.linear(batch.x.view(batch.num_graphs, -1))

    def shared_step(self, batch: Data, step_name: str) -> dict:
        """Shared step for training and validation."""
        y_hat = self.forward(batch).squeeze(-1)
        y = (batch.y > 2.5).long()
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        acc = metrics.accuracy(y_hat, y, "binary")
        auc = metrics.auroc(y_hat, y, "binary")
        mcc = metrics.matthews_corrcoef(y_hat, y, "binary")
        self.log(f"{step_name}/loss", loss, batch_size=batch.num_graphs)
        self.log(f"{step_name}/acc", acc, batch_size=batch.num_graphs)
        self.log(f"{step_name}/auc", auc, batch_size=batch.num_graphs)
        self.log(f"{step_name}/mcc", mcc, batch_size=batch.num_graphs)
        if self.global_step % 100 == 0:
            wandb.log(
                {
                    f"{step_name}/conf_mat": wandb.plot.confusion_matrix(
                        preds=(y_hat > 0).long().detach().cpu().numpy(), y_true=y.long().detach().cpu().numpy()
                    )
                }
            )
        return dict(loss=loss, acc=acc, auc=auc, mcc=mcc)

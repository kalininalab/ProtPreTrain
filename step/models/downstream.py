from typing import Any

import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

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

    def forward(self, batch: Data) -> torch.Tensor:
        """Use the simple MLP."""
        x, _ = to_dense_batch(batch.x, batch.batch)
        return self.linear(x)

    def training_step(self, *args, **kwargs) -> dict:
        """Training step."""
        if self.global_step == 0:
            wandb.define_metric("val/loss", summary="min")
        return self.shared_step(*args, **kwargs, step_name="train")

    def validation_step(self, *args, **kwargs) -> dict:
        """Validation step."""
        return self.shared_step(*args, **kwargs, step_name="val")

    def test_step(self, *args, **kwargs) -> dict:
        """Validation step."""
        return self.shared_step(*args, **kwargs, step_name="test")

    def configure_optimizers(self) -> Any:
        """Configure optimizers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
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

    def shared_step(self, batch: Data, batch_idx: int = 0, *, step_name: str = "train") -> dict:
        """Shared step for training and validation."""
        y_hat = self.forward(batch).squeeze(-1)
        y = batch.y
        loss = F.mse_loss(y_hat, y)
        mae = metrics.mean_absolute_error(y_hat, y)
        r2 = metrics.r2_score(y_hat, y)
        spear = metrics.spearman_corrcoef(y_hat, y)
        pears = metrics.pearson_corrcoef(y_hat, y)
        metrics_dict = {
            f"{step_name}/loss": loss,
            f"{step_name}/mae": mae,
            f"{step_name}/r2": r2,
            f"{step_name}/spearman": spear,
            f"{step_name}/pearson": pears,
        }
        self.log_dict(metrics_dict, batch_size=batch.num_graphs, add_dataloader_idx=False)
        return dict(loss=loss, mae=mae, r2=r2, spear=spear, pears=pears)


class ClassificationModel(BaseModel):
    """Takes in precomputed embeddings and predicts a class"""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.linear = LazySimpleMLP(hidden_dim, num_classes, dropout)
        self.num_classes = num_classes

    def shared_step(self, batch: Data, batch_idx: int = 0, *, step_name: str = "train") -> dict:
        """Shared step for training and validation."""
        y_hat = self.forward(batch)
        y = torch.tensor(batch.y, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(y_hat, y)
        acc = metrics.accuracy(y_hat, y, "multiclass", num_classes=self.num_classes)
        auc = metrics.auroc(y_hat, y, "multiclass", num_classes=self.num_classes)
        mcc = metrics.matthews_corrcoef(y_hat, y, "multiclass", num_classes=self.num_classes)
        metrics_dict = {
            f"{step_name}/loss": loss,
            f"{step_name}/acc": acc,
            f"{step_name}/auc": auc,
            f"{step_name}/mcc": mcc,
        }
        self.log_dict(metrics_dict, batch_size=batch.num_graphs, add_dataloader_idx=False)
        return dict(loss=loss, acc=acc, auc=auc, mcc=mcc)


class HomologyModel(ClassificationModel):
    def test_step(self, batch: Data, batch_idx: int, dataloader_idx: int) -> dict:
        """Test step."""
        step_name = ["fold", "superfamily", "family"]
        return self.shared_step(batch, step_name=f"test_{step_name[dataloader_idx]}")


class DTIModel(RegressionModel):
    """ECFP + Protein sequence -> Binding affinity"""

    def __init__(
        self,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.linear_prot = LazySimpleMLP(hidden_dim, hidden_dim, dropout)
        self.linear_drug = LazySimpleMLP(1024, hidden_dim, dropout)
        self.linear = LazySimpleMLP(hidden_dim * 2, 1, dropout)

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        prot, _ = to_dense_batch(batch.x, batch.batch)
        drug, _ = to_dense_batch(batch.ecfp, batch.batch)
        prot = self.linear_prot(prot)
        drug = self.linear_drug(drug)
        x = torch.cat([prot, drug], dim=1)
        return self.linear(x)

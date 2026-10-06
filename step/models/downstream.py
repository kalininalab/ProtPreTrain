from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassMatthewsCorrCoef
from torchmetrics.regression import MeanAbsoluteError, PearsonCorrCoef, R2Score, SpearmanCorrCoef


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


def make_head(head: str, hidden_dim: int, out_dim: int, dropout: float) -> torch.nn.Module:
    """Output head on frozen embeddings: ``mlp`` (one hidden layer) or ``linear`` (a linear probe)."""
    if head == "mlp":
        return LazySimpleMLP(hidden_dim, out_dim, dropout)
    if head == "linear":
        return torch.nn.LazyLinear(out_dim)
    raise ValueError(f"unknown head {head!r}, expected 'mlp' or 'linear'")


class BaseModel(LightningModule):
    """Base class for all downstream stuff.

    Metrics are torchmetrics objects per stage, accumulated over the whole epoch and computed once at its end:
    correlations, R2, AUROC and MCC are not averages of per-batch values (which differ from the dataset-level metric,
    badly so when a split is sorted by label or target).
    """

    stages = ("train", "val", "test")

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def _stage_metrics(self, make: dict) -> torch.nn.ModuleDict:
        """One MetricCollection per stage, logged as ``<stage>/<name>``; ``make`` maps names to metric factories.

        Keyed ``<stage>_metrics``: a ModuleDict key cannot be ``train``, which would shadow ``nn.Module.train``.
        """
        return torch.nn.ModuleDict(
            {
                f"{stage}_metrics": MetricCollection({k: f() for k, f in make.items()}, prefix=f"{stage}/")
                for stage in self.stages
            }
        )

    def stage_metrics(self, stage: str) -> MetricCollection:
        """The metrics accumulated for ``stage`` (e.g. ``"val"`` or ``"test_fold"``)."""
        return self.epoch_metrics[f"{stage}_metrics"]

    def _log_epoch_metrics(self, step_name: str, preds: torch.Tensor, target: torch.Tensor, batch_size: int):
        """Accumulate this batch into the stage's metrics; Lightning computes and resets them at epoch end."""
        collection = self.stage_metrics(step_name)
        collection.update(preds, target)
        self.log_dict(collection, on_step=False, on_epoch=True, batch_size=batch_size, add_dataloader_idx=False)

    def forward(self, batch: Data) -> torch.Tensor:
        """Use the simple MLP."""
        x, _ = to_dense_batch(batch.x, batch.batch)
        return self.linear(x)

    def training_step(self, *args, **kwargs) -> dict:
        """Training step."""
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
        )
        return [optimizer], [{"scheduler": scheduler, "monitor": "val/loss"}]


class RegressionModel(BaseModel):
    """Base for regression, only defines shared steps"""

    def __init__(
        self,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        head: str = "mlp",
    ):
        super().__init__()
        self.linear = make_head(head, hidden_dim, 1, dropout)
        self.epoch_metrics = self._stage_metrics(
            {"mae": MeanAbsoluteError, "r2": R2Score, "spearman": SpearmanCorrCoef, "pearson": PearsonCorrCoef}
        )

    def shared_step(self, batch: Data, batch_idx: int = 0, *, step_name: str = "train") -> dict:
        """Shared step for training and validation."""
        y_hat = self.forward(batch).squeeze(-1)
        y = batch.y
        loss = F.mse_loss(y_hat, y)
        self.log(f"{step_name}/loss", loss, batch_size=batch.num_graphs, add_dataloader_idx=False)
        self._log_epoch_metrics(step_name, y_hat.float(), y.float(), batch.num_graphs)
        return dict(loss=loss)


class ClassificationModel(BaseModel):
    """Takes in precomputed embeddings and predicts a class"""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        head: str = "mlp",
    ):
        super().__init__()
        self.linear = make_head(head, hidden_dim, num_classes, dropout)
        self.num_classes = num_classes
        self.epoch_metrics = self._stage_metrics(
            {
                # micro: plain fraction correct, as the functional accuracy logged before
                "acc": lambda: MulticlassAccuracy(num_classes, average="micro"),
                # weighted by class support: classes absent from a split (most of homology's 1195 in any test set)
                # count 0 instead of dragging a macro average towards zero
                "auc": lambda: MulticlassAUROC(num_classes, average="weighted"),
                "mcc": lambda: MulticlassMatthewsCorrCoef(num_classes),
            }
        )

    def shared_step(self, batch: Data, batch_idx: int = 0, *, step_name: str = "train") -> dict:
        """Shared step for training and validation."""
        y_hat = self.forward(batch)
        y = torch.as_tensor(batch.y, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(y_hat, y)
        self.log(f"{step_name}/loss", loss, batch_size=batch.num_graphs, add_dataloader_idx=False)
        self._log_epoch_metrics(step_name, y_hat.float(), y, batch.num_graphs)
        return dict(loss=loss)


class HomologyModel(ClassificationModel):
    """Fold classification with three test sets (fold / superfamily / family holdout), each with its own metrics."""

    stages = ("train", "val", "test_fold", "test_superfamily", "test_family")

    def test_step(self, batch: Data, batch_idx: int, dataloader_idx: int) -> dict:
        """Test step."""
        step_name = ["fold", "superfamily", "family"]
        return self.shared_step(batch, step_name=f"test_{step_name[dataloader_idx]}")


class DTIModel(RegressionModel):
    """ECFP + Protein sequence -> Binding affinity"""

    def __init__(
        self,
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.linear_prot = LazySimpleMLP(hidden_dim, hidden_dim, dropout)
        self.linear_drug = LazySimpleMLP(1024, hidden_dim, dropout)
        self.linear = LazySimpleMLP(hidden_dim * 2, 1, dropout)

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        num_nodes = len(batch)
        prot = batch.x.view(num_nodes, -1)
        drug = batch.ecfp.view(num_nodes, -1)
        prot = self.linear_prot(prot)
        drug = self.linear_drug(drug)
        x = torch.cat([prot, drug], dim=1)
        return self.linear(x)

import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torchmetrics.functional as metrics
from graphgps.layer.gps_layer import GPSLayer
from pytorch_lightning import LightningModule
from torch.nn.utils import weight_norm
from torch_geometric.data import Data

import wandb


class SimpleMLP(torch.nn.Module):
    """Simple MLP for output heads"""

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.main = torch.nn.Sequential(
            weight_norm(torch.nn.Linear(in_dim, hid_dim), dim=None),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            weight_norm(torch.nn.Linear(hid_dim, out_dim), dim=None),
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


class RegressionBaseModel(BaseModel):
    """Base for regression, only defines shared steps"""

    def shared_step(self, batch: Data, step_name: str) -> dict:
        """Shared step for training and validation."""
        y = self.forward(batch).squeeze(-1)
        pred_y = batch.y
        loss = F.huber_loss(pred_y, y, delta=0.5)
        mae = metrics.mean_absolute_error(pred_y, y)
        r2 = metrics.r2_score(pred_y, y)
        corrcoef = metrics.spearman_corrcoef(pred_y, y)
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


class RegressionModel(RegressionBaseModel):
    """Uses GraphGPS transformer layers to encode the graph and predict the Y value."""

    def __init__(
        self,
        local_module: str = "GAT",
        global_module: str = "Performer",
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_encode = torch.nn.Embedding(21, hidden_dim)
        # self.edge_encode = torch.nn.Linear(3, hidden_dim)
        self.node_encode = torch.nn.ModuleList(
            [
                # GPSLayer(
                #     hidden_dim,
                #     local_module,
                #     global_module,
                #     num_heads,
                #     dropout=dropout,
                #     attn_dropout=attn_dropout,
                # )
                pyg.nn.GCNConv(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )
        # self.aggr = pyg.nn.aggr.SetTransformerAggregation(
        #     hidden_dim,
        #     num_encoder_blocks=2,
        #     num_decoder_blocks=2,
        #     heads=4,
        #     dropout=dropout,
        # )
        self.aggr = pyg.nn.MeanAggregation()
        self.linear = SimpleMLP(hidden_dim, hidden_dim, 1, dropout)

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        batch.x = self.feat_encode(batch.x)
        # batch.edge_attr = self.edge_encode(batch.edge_attr)
        for layer in self.node_encode:
            batch.x = layer(batch.x, batch.edge_index)
        x = self.aggr(batch.x, batch.batch)
        return self.linear(x)


class RegressionESMModel(RegressionBaseModel):
    """Uses ESM."""

    def __init__(
        self,
        in_dim: int = 1280,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.linear = SimpleMLP(in_dim, hidden_dim, 1, dropout)

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        return self.linear(batch.x.view(batch.num_graphs, -1))


class MDNESMModel(RegressionBaseModel):
    """Uses ESM and MDN."""

    def __init__(
        self,
        in_dim: int = 1280,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.linear = SimpleMLP(in_dim, hidden_dim, 1, dropout)

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        return self.linear(batch.x.view(batch.num_graphs, -1))

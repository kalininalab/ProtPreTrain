import torch
import torch.nn.functional as F
from graphgps.layer.gps_layer import GPSLayer
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
import torchmetrics.functional as metrics
from torch_geometric import nn
import wandb

class RegressionModel(LightningModule):
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
        self.aggr = nn.aggr.SetTransformerAggregation(
            hidden_dim, num_encoder_blocks=4, num_decoder_blocks=4, heads=4, dropout=dropout
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Data) -> Data:
        """Return updated batch with noise and node type predictions."""
        batch.x = self.feat_encode(batch.x)
        batch.edge_attr = self.edge_encode(batch.edge_attr)
        batch = self.node_encode(batch)
        x = self.aggr(batch.x, batch.batch)
        return self.linear(x)

    def shared_step(self, batch: Data, step_name: str) -> dict:
        """Shared step for training and validation."""
        y = self.forward(batch)
        pred_y = batch.y.unsqueeze(-1)
        loss = F.mse_loss(pred_y, y)
        mae = metrics.mean_absolute_error(pred_y, y)
        r2 = metrics.r2_score(pred_y, y)
        self.log(f"{step_name}/loss", loss,batch_size=batch.num_graphs)
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
            wandb.define_metric('val/loss', summary='min')
        return self.shared_step(batch, "train")

    def validation_step(self, batch: Data, batch_idx: int) -> dict:
        """Validation step."""
        return self.shared_step(batch, "val")

    def test_step(self, batch: Data, batch_idx: int) -> dict:
        """Test step."""
        return self.shared_step(batch, "test")

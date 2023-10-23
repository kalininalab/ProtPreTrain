import argparse
import warnings

import pytorch_lightning as pl
import torch
import torch_geometric.transforms as T

import wandb
from step.data import FluorescenceDataModule, StabilityDataModule
from step.models import RegressionModel

# Ignore all deprecation warnings
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="fluorescence", choices=["fluorescence", "stability"])
parser.add_argument("--model_source", type=str, choices=["wandb", "huggingface", "ankh"])
parser.add_argument("--model", type=str)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--num_workers", type=int, default=0)
config = parser.parse_args()

logger = pl.loggers.WandbLogger(project="STEP_" + config.dataset, log_model=True, dir="wandb", config=config)

config = wandb.config
print(config)

model = RegressionModel(hidden_dim=config.hidden_dim, dropout=config.dropout)
data = {
    "fluorescence": FluorescenceDataModule,
    "stability": StabilityDataModule,
}[config.dataset](
    feature_extract_model=config.model,
    feature_extract_model_source=config.model_source,
    num_workers=config.num_workers,
    batch_size=config.batch_size,
)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    precision="bf16",
    max_epochs=10000,
    logger=logger,
    callbacks=[
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", dirpath="checkpoints"),
        pl.callbacks.RichProgressBar(),
        pl.callbacks.RichModelSummary(),
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=50, mode="min"),
    ],
)
trainer.fit(model, data)
trainer.test(model, data)
wandb.finish()

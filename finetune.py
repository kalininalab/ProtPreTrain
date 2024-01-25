import argparse
import warnings

import pytorch_lightning as pl
import torch

import wandb
from step.data import FluorescenceDataModule, HomologyDataModule, StabilityDataModule
from step.data.datamodules import DTIDataModule
from step.models import HomologyModel, RegressionModel, DTIModel

# Ignore all deprecation warnings
torch.set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy("file_system")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="fluorescence", choices=["fluorescence", "stability", "homology", "dti"])
parser.add_argument("--model_source", type=str, choices=["wandb", "huggingface", "ankh", "prostt5"])
parser.add_argument("--model", type=str)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--ablation", type=str, default="none", choices=["none", "sequence", "structure"])
config = parser.parse_args()

logger = pl.loggers.WandbLogger(project=config.dataset, log_model=True, dir="wandb", config=config, entity="rindti")

config = wandb.config
print(config)
if config.dataset == "homology":
    model = HomologyModel(hidden_dim=config.hidden_dim, dropout=config.dropout, num_classes=1195)
elif config.dataset == "dti":
    model = DTIModel(hidden_dim=config.hidden_dim, dropout=config.dropout)
else:
    model = RegressionModel(hidden_dim=config.hidden_dim, dropout=config.dropout)
data = {
    "fluorescence": FluorescenceDataModule,
    "stability": StabilityDataModule,
    "homology": HomologyDataModule,
    "dti": DTIDataModule,
}[config.dataset](
    feature_extract_model=config.model,
    feature_extract_model_source=config.model_source,
    num_workers=config.num_workers,
    batch_size=config.batch_size,
    ablation=config.ablation,
)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    precision="bf16-mixed",
    max_epochs=10000,
    logger=logger,
    callbacks=[
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", dirpath="checkpoints"),
        pl.callbacks.RichProgressBar(),
        pl.callbacks.RichModelSummary(),
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=10, mode="min"),
    ],
)
trainer.fit(model, data)
trainer.test(model, data)

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
for dataset in ["fluorescence", "stability"]:
    for model, model_source in {
        "ilsenatorov/step/model-2qss8cgz:v0": "wandb",
        "ankh-base": "ankh",
        "facebook/esm2_t6_8M_UR50D": "huggingface",
        "Rostlab/prot_bert": "huggingface",
    }.items():
        config = {
            "dataset": dataset,  # "fluorescence", "stability"
            "model_source": model_source,  # "wandb", "huggingface", "ankh"
            "model": model,
            "hidden_dim": 512,
            "dropout": 0.2,
        }

        model = RegressionModel(hidden_dim=config["hidden_dim"], dropout=config["dropout"])
        data = {
            "fluorescence": FluorescenceDataModule,
            "stability": StabilityDataModule,
        }[config["dataset"]](
            feature_extract_model=config["model"],
            feature_extract_model_source=config["model_source"],
            num_workers=8,
            batch_size=256,
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=-1,
            precision="bf16",
            max_epochs=10000,
            logger=pl.loggers.WandbLogger(project=config["dataset"], log_model=True, dir="wandb", config=config),
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

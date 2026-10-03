import argparse
import json
import warnings

import lightning.pytorch as pl
import torch

from step.data import FluorescenceDataModule, HomologyDataModule, StabilityDataModule
from step.data.datamodules import DTIDataModule
from step.models import DTIModel, HomologyModel, RegressionModel
from step.utils import mlflow_logger, progress_bar

# Ignore all deprecation warnings
torch.set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy("file_system")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="fluorescence", choices=["fluorescence", "stability", "homology", "dti"]
)
parser.add_argument("--model_source", type=str, choices=["checkpoint", "huggingface", "ankh", "prostt5"])
parser.add_argument("--model", type=str, help="Local .ckpt path (checkpoint) or model name")
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--ablation", type=str, default="none", choices=["none", "sequence", "structure"])
parser.add_argument(
    "--ablation_maskfrac", type=float, default=1.0, help="Fraction of residues masked by --ablation structure"
)
parser.add_argument(
    "--random_init", action="store_true", help="No-pretraining control: reinitialise the checkpoint model's weights"
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_epochs", type=int, default=10000)
parser.add_argument("--experiment", type=str, default=None, help="MLflow experiment name; defaults to the dataset")
parser.add_argument(
    "--embed_cache",
    type=str,
    default=None,
    help="Directory for per-split embeddings; reused if present, so head seeds embed only once per encoder",
)
parser.add_argument("--summary_json", type=str, default=None, help="Write the test metrics here as JSON")
config = parser.parse_args()
pl.seed_everything(config.seed)

logger = mlflow_logger(config.experiment or config.dataset)
logger.log_hyperparams(vars(config))
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
    ablation_maskfrac=config.ablation_maskfrac,
    random_init=config.random_init,
    embed_cache=config.embed_cache,
)
trainer = pl.Trainer(
    accelerator="auto",
    devices="auto",
    # CPUs without native bf16 run bf16 autocast several times slower than fp32
    precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
    max_epochs=config.max_epochs,
    logger=logger,
    callbacks=[
        # Per-run directory: concurrent jobs sharing one would overwrite each other's identically named checkpoints
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", dirpath=f"checkpoints/{logger.run_id}"),
        progress_bar(),
        pl.callbacks.RichModelSummary(),
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=10, mode="min"),
    ],
)
trainer.fit(model, data)
results = trainer.test(model, data, ckpt_path="best")
if config.summary_json:
    with open(config.summary_json, "w") as f:
        json.dump({k: v for r in results for k, v in r.items()}, f, indent=2)

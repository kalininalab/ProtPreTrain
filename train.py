import torch
from pytorch_lightning.cli import LightningCLI

from step.data import FoldSeekDataModule, FoldSeekSmallDataModule
from step.models.denoise import DenoiseModel
from step.utils.cli import namespace_to_dict
import wandb
import warnings

# Ignore all deprecation warnings
warnings.filterwarnings("ignore")


torch.set_float32_matmul_precision("medium")
cli = LightningCLI(
    model_class=DenoiseModel,
    datamodule_class=FoldSeekSmallDataModule,
    save_config_callback=None,
    run=False,
)
wandb.init(
    project="step",
    name="foldseek",
    config=namespace_to_dict(cli.config),
)
model = cli.model
# model = torch_geometric.compile(model)
cli.trainer.fit(model, cli.datamodule)

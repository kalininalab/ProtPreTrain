import wandb
import torch
from pytorch_lightning.cli import LightningCLI

from step.data import AlphaFoldDataModule
from step.models.denoise import DenoiseModel
from step.utils.cli import namespace_to_dict

torch.set_float32_matmul_precision("medium")
cli = LightningCLI(
    model_class=DenoiseModel,
    datamodule_class=AlphaFoldDataModule,
    run=False,
    save_config_callback=None,
)
wandb.init(
    project="pretrain_alphafold",
    name="swissprot",
    config=namespace_to_dict(cli.config),
)
cli.trainer.fit(cli.model, cli.datamodule)

import torch
from pytorch_lightning.cli import LightningCLI

from step.data import FoldSeekDataModule
from step.models.denoise import DenoiseModel


torch.set_float32_matmul_precision('medium')
cli = LightningCLI(model_class=DenoiseModel, datamodule_class=FoldSeekDataModule, save_config_callback=None)

import warnings

import torch
from pytorch_lightning.cli import LightningCLI, ReduceLROnPlateau

from step.data import FluorescenceDataModule, FoldSeekDataModule, StabilityDataModule
from step.models import DenoiseModel, RegressionModel
from step.utils.cli import namespace_to_dict

# Ignore all deprecation warnings
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

cli = LightningCLI(
    run=False,
    save_config_callback=None,
)

model = cli.model
datamodule = cli.datamodule
datamodule.setup()
cli.trainer.logger.experiment.config.update(namespace_to_dict(cli.config))
cli.trainer.fit(model, datamodule=datamodule)
cli.trainer.test(model, datamodule=datamodule)

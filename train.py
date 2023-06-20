import torch
from pytorch_lightning.cli import LightningCLI


from step.data import (
    FluorescenceDataModule,
    StabilityDataModule,
    FoldSeekDataModule,
    FoldSeekSmallDataModule,
)
from step.models import RegressionModel, DenoiseModel
import warnings
from step.utils.cli import namespace_to_dict


# Ignore all deprecation warnings
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

cli = LightningCLI(
    save_config_callback=None,
    run=False,
)

model = cli.model
datamodule = cli.datamodule
datamodule.setup()
cli.trainer.logger.experiment.config.update(namespace_to_dict(cli.config))
cli.trainer.fit(model, datamodule=datamodule)

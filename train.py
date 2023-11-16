import warnings

import torch
from pytorch_lightning.cli import LightningCLI
import wandb

<<<<<<< HEAD
from step.data import FluorescenceDataModule, FoldSeekDataModule, StabilityDataModule
from step.models import DenoiseModel, RegressionModel
=======
>>>>>>> b3ae6fbc972f22114b63f7a609a07ab54bb0dc04
from step.utils.cli import namespace_to_dict

torch.set_float32_matmul_precision("medium")
<<<<<<< HEAD
torch.multiprocessing.set_sharing_strategy("file_system")
=======
wandb.init(settings=wandb.Settings(start_method="fork"), project="STEP", name="test", entity="rindti")
>>>>>>> b3ae6fbc972f22114b63f7a609a07ab54bb0dc04

cli = LightningCLI(
    run=False,
    save_config_callback=None,
)

model = cli.model
datamodule = cli.datamodule
datamodule.setup()
try:
    cli.trainer.logger.experiment.config.update(namespace_to_dict(cli.config))
except:
    pass
cli.trainer.fit(model, datamodule=datamodule)
cli.trainer.test(model, datamodule=datamodule)

import torch
from pytorch_lightning.cli import LightningCLI
import wandb
from step.utils.cli import namespace_to_dict

torch.set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy("file_system")
wandb.init(settings=wandb.Settings(start_method="fork"), project="step", entity="rindti", mode="offline")

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

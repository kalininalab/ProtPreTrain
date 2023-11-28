import pytorch_lightning as pl
import torch

import wandb
from step.utils.cli import namespace_to_dict

pl.seed_everything(42)

torch.set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy("file_system")

cli = pl.cli.LightningCLI(run=False, save_config_callback=None)

cli.trainer.logger = pl.loggers.WandbLogger(
    experiment=wandb.run,
    offline=True,
    project="step",
    entity="rindti",
    settings=wandb.Settings(start_method="fork"),
    config=namespace_to_dict(cli.config),
)
model = cli.model
datamodule = cli.datamodule
datamodule.setup()

cli.trainer.fit(model, datamodule=datamodule)
# cli.trainer.test(model, datamodule=datamodule)

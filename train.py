from pytorch_lightning.cli import LightningCLI

from protpretrain.data import AlphaFoldDataModule
from protpretrain.models.denoise import DenoiseModel
from protpretrain.utils.cli import namespace_to_dict

if __name__ == "__main__":
    cli = LightningCLI(
        model_class=DenoiseModel,
        datamodule_class=AlphaFoldDataModule,
        run=False,
        save_config_callback=None,
    )
    cli.trainer.logger.experiment.config.update(namespace_to_dict(cli.config))
    cli.trainer.fit(cli.model, cli.datamodule)

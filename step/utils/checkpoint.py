import pytorch_lightning as pl
import wandb


class WandbArtifactModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, wandb_run, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb_run = wandb_run

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str):
        # Call the original save method to save the checkpoint file
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero and self.wandb_run:
            artifact = wandb.Artifact(
                name=f"model-{self.wandb_run.id}",
                type="model",
                metadata=dict(trainer.callback_metrics),
            )
            artifact.add_file(filepath)
            self.wandb_run.log_artifact(artifact)
            # print(f"Logged {checkpoint_path} as a W&B artifact.")

import glob
import json
import os
from collections import deque

import lightning.pytorch as pl
import torch

from .cli import resolve_keep_steps

# Every kept checkpoint is validated with the same noise and masks, so its val losses differ only by the weights
VAL_SEED = 0


def kept_name(step: int) -> str:
    """File stem of the checkpoint kept at optimizer step ``step``."""
    return f"step_{step:07d}"


def read_kept(dirpath: str) -> list:
    """Sidecar records (one per kept checkpoint) under ``dirpath``, sorted by step."""
    records = [json.load(open(p)) for p in glob.glob(os.path.join(dirpath, "step_*.json"))]
    return sorted(records, key=lambda r: r["step"])


class KeepCheckpoints(pl.Callback):
    """Keep weights-only checkpoints at chosen optimizer steps, each with its validation losses.

    At every step in ``tokens`` (see ``resolve_keep_steps``; resolved against the run's total steps) this writes
    ``<dirpath>/step_<step>.ckpt`` and a sidecar ``step_<step>.json`` holding the step, epoch, learning rate, the
    validation losses of exactly those weights (``val/loss``, ``val/noise_loss``, ``val/pred_loss``, ``val/pred_acc``)
    and the mean training loss over the last ``window`` steps. The same values go to the logger as ``kept/...`` at
    that step. Validation runs on rank 0 over the whole val loader, with fixed noise (``VAL_SEED``) and in eval mode;
    the global RNG state is restored afterwards, so training is unaffected.

    Restart-safe: kept steps and the loss window travel in ``last.ckpt`` through ``state_dict``, so a resumed run
    keeps only steps it has not reached yet; a step reached again after a crash overwrites its earlier files, which
    keeps checkpoint and sidecar from the same weights.
    """

    def __init__(self, dirpath: str, tokens: list, window: int = 200, val_loader_fn=None):
        super().__init__()
        self.dirpath = dirpath
        self.tokens = list(tokens)
        self.window = deque(maxlen=window)
        self.val_loader_fn = val_loader_fn
        self.steps = []
        self.total = None
        self.saved = set()

    def state_dict(self) -> dict:
        """Kept steps and the training-loss window, stored in every full checkpoint."""
        return {"saved": sorted(self.saved), "window": list(self.window)}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore what a resumed run has already kept."""
        self.saved = set(state_dict.get("saved", []))
        self.window.extend(state_dict.get("window", []))

    def on_train_start(self, trainer, pl_module) -> None:
        """Resolve the schedule against the run length; step 0 is kept before the first update."""
        self.total = int(trainer.estimated_stepping_batches)
        self.steps = resolve_keep_steps(self.tokens, self.total)
        missed = [s for s in self.steps if s < trainer.global_step and s not in self.saved]
        if missed:
            pl.utilities.rank_zero_warn(f"resumed at step {trainer.global_step}; can't keep earlier steps {missed}")
        if trainer.global_step == 0 and 0 in self.steps:
            self._keep(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Track the training loss and keep a checkpoint when the step is on the schedule."""
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if loss is not None:
            self.window.append(float(loss.detach()))
        if trainer.global_step in self.steps and trainer.global_step not in self.saved:
            self._keep(trainer, pl_module)

    def on_train_end(self, trainer, pl_module) -> None:
        """A run that stops before its planned last step (e.g. max_time) still keeps ``final``."""
        if "final" in self.tokens and trainer.global_step not in self.saved and not trainer.interrupted:
            self._keep(trainer, pl_module)

    @torch.no_grad()
    def validate(self, trainer, pl_module) -> dict:
        """Graph-weighted mean val losses of the current weights over the whole val loader (``{}`` without one)."""
        loader = self.val_loader_fn() if self.val_loader_fn else None
        if loader is None and trainer.datamodule is not None:
            loader = trainer.datamodule.val_dataloader()
        if not loader:
            return {}
        was_training = pl_module.training
        pl_module.eval()
        totals, n = {}, 0
        # manual_seed also reseeds CUDA, so fork this rank's GPU too (only this one: others may belong to other ranks)
        devices = [pl_module.device.index or 0] if pl_module.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), trainer.strategy.precision_plugin.val_step_context():
            torch.manual_seed(VAL_SEED)  # dataloader worker seeds are drawn from here
            for batch in loader:
                batch = batch.to(pl_module.device)
                for k, v in pl_module.losses(batch).items():
                    totals[k] = totals.get(k, 0.0) + float(v) * batch.num_graphs
                n += batch.num_graphs
        pl_module.train(was_training)
        return {f"val/{k}": v / n for k, v in totals.items()}

    def _keep(self, trainer, pl_module) -> None:
        step = trainer.global_step
        path = os.path.join(self.dirpath, f"{kept_name(step)}.ckpt")
        # save_checkpoint is collective under DDP; every rank calls it, rank 0 writes
        trainer.save_checkpoint(path, weights_only=True)
        if trainer.is_global_zero:
            record = {
                "step": step,
                "epoch": trainer.current_epoch,
                "planned_steps": self.total,
                "ckpt_path": os.path.abspath(path),
                "lr": trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else None,
                "train/loss_window": sum(self.window) / len(self.window) if self.window else None,
                "train/window_steps": len(self.window),
                **self.validate(trainer, pl_module),
            }
            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {f"kept/{k}": v for k, v in record.items() if "/" in k and v is not None}, step=step
                )
            tmp = os.path.join(self.dirpath, f".{kept_name(step)}.json.tmp")
            with open(tmp, "w") as f:
                json.dump(record, f, indent=2)
            os.replace(tmp, os.path.join(self.dirpath, f"{kept_name(step)}.json"))  # sidecar last: marks complete
        trainer.strategy.barrier("KeepCheckpoints")
        self.saved.add(step)

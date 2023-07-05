import math

import torch


class WarmUpCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup + Cosine LR. Uses number of steps rather than number of epochs.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 100,
        start_lr: float = 1.0e-5,
        max_lr: float = 1.0e-4,
        min_lr: float = 1.0e-7,
        cycle_len: int = 1000,
    ):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_len = cycle_len
        self.step_count: int = 0
        super().__init__(optimizer)

    def _get_warmup_lr(self):
        lr = self.start_lr + (self.max_lr - self.start_lr) * self.step_count / self.warmup_steps
        return lr

    def _get_cosine_lr(self):
        lr = (
            self.min_lr
            + (self.max_lr - self.min_lr)
            * (1 + math.cos(math.pi * (self.step_count - self.warmup_steps) / self.cycle_len))
            / 2
        )
        return lr

    def get_lr(self):
        """Calculates lr. Linear before `warmup_steps`, cosine between `max_lr` and `min_lr` after."""
        if self.step_count <= self.warmup_steps:
            return [self._get_warmup_lr() for _ in self.optimizer.param_groups]
        # elif self.step_count >= self.warmup_steps + self.cycle_len:
        #     return [self.min_lr for _ in self.optimizer.param_groups]
        else:
            return [self._get_cosine_lr() for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        """Updates number of steps, then same stuff."""
        self.step_count += 1
        super().step(epoch)

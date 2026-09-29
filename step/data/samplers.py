from typing import Iterator, List

import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler


class DynamicBatchSampler(Sampler[List[int]]):
    """Batch indices so that each batch holds at most max_num nodes, sharded evenly across DDP ranks.

    Batching is decided from precomputed lengths, so no sample is loaded to size it. Every rank builds the same batch
    list from a shared per-epoch seed and takes every world_size-th batch, truncated so all ranks run the same number
    of steps (DDP hangs otherwise). Use with use_distributed_sampler=False; Lightning calls set_epoch.
    """

    def __init__(self, lengths: np.ndarray, max_num: int, shuffle: bool = True, seed: int = 0):
        self.lengths = np.asarray(lengths)
        self.max_num = max_num
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if distributed else 0
        self.world_size = dist.get_world_size() if distributed else 1
        self._cache = None

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle for a new epoch."""
        self.epoch = epoch

    def _batches(self) -> List[List[int]]:
        if self._cache is not None and self._cache[0] == self.epoch:
            return self._cache[1]
        n = len(self.lengths)
        order = np.random.default_rng(self.seed + self.epoch).permutation(n) if self.shuffle else np.arange(n)
        batches, batch, num = [], [], 0
        for i, size in zip(order.tolist(), self.lengths[order].tolist()):
            if batch and num + size > self.max_num:
                batches.append(batch)
                batch, num = [], 0
            batch.append(i)
            num += size
        if batch:
            batches.append(batch)
        steps = len(batches) // self.world_size
        batches = batches[self.rank : steps * self.world_size : self.world_size]
        self._cache = (self.epoch, batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())

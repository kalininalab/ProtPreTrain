import os
import shutil
import sys
import timeit

import torch_geometric.transforms as T

import wandb
from step.data import FoldCompDataset, RandomWalkPE

wandb.init(project="data_processing", entity="rindti")

ds = FoldCompDataset(
    db_name="afdb_rep_v4",
    pre_transform=T.Compose(
        [
            T.Center(),
            T.NormalizeRotation(),
            T.RadiusGraph(10),
            T.ToUndirected(),
            RandomWalkPE(20, "pe", cuda=True),
        ]
    ),
    num_workers=args.num_workers,
    chunk_size=args.chunk_size,
)

print(f"Length: {len(ds)}")

number = 10
len_time = timeit.timeit(lambda: len(ds), number=number) / number * 1000
get_time = timeit.timeit(lambda: ds[0], number=number) / number * 1000
print(f"Len time: {len_time}ms, Get time: {get_time}ms")

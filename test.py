import os
import shutil
import sys
import timeit

import torch_geometric.transforms as T

import wandb
from step.data import FoldCompDataset, RandomWalkPE

ds_name = "afdb_rep_v4"
if os.path.exists(f"data/{ds_name}/processed"):
    shutil.rmtree(f"data/{ds_name}/processed")

ds = FoldCompDataset(
    db_name=ds_name,
    pre_transform=T.Compose(
        [
            T.Center(),
            T.NormalizeRotation(),
            T.RadiusGraph(10),
            T.ToUndirected(),
            RandomWalkPE(20, "pe", cuda=True),
        ]
    ),
    num_workers=8,
    chunk_size=64,
)

print(f"Length: {len(ds)}")

number = 10
len_time = timeit.timeit(lambda: len(ds), number=number) / number * 1000
get_time = timeit.timeit(lambda: ds[0], number=number) / number * 1000
print(f"Len time: {len_time}ms, Get time: {get_time}ms")

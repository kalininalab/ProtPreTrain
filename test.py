import os
import shutil
import sys
import timeit

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from step.data import FoldCompDataset, RandomWalkPE

ds_name = "e_coli"
if os.path.exists(f"data/{ds_name}/processed"):
    shutil.rmtree(f"data/{ds_name}/processed")

ds = FoldCompDataset(
    db_name=ds_name,
    schema={
        "x": dict(dtype=torch.int64, size=(-1,)),
        "edge_index": dict(dtype=torch.int64, size=(2, -1)),
        "pos": dict(dtype=torch.float32, size=(-1, 3)),
        "pe": dict(dtype=torch.float32, size=(-1, 20)),
        "uniprot_id": float,
    },
    pre_transform=T.Compose(
        [
            T.Center(),
            T.NormalizeRotation(),
            T.RadiusGraph(10),
            T.ToUndirected(),
            RandomWalkPE(20, "pe", cuda=True),
        ]
    ),
    num_workers=14,
    chunk_size=2048,
)

print(f"Length: {len(ds)}")
dl = DataLoader(ds, batch_size=32, num_workers=14)

number = 100
len_time = timeit.timeit(lambda: len(ds), number=number) / number * 1000
get_time = timeit.timeit(lambda: ds[0], number=number) / number * 1000
dl_time = timeit.timeit(lambda: next(iter(dl)), number=number) / number * 1000
print(f"Len time: {len_time}ms, Get time: {get_time}ms, DL time: {dl_time}ms")

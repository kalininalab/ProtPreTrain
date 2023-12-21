import os
import shutil

import torch_geometric.transforms as T

from step.data import FoldSeekDatasetSmall, RandomWalkPE

if os.path.exists("data/foldseek_small/processed"):
    shutil.rmtree("data/foldseek_small/processed")

ds = FoldSeekDatasetSmall(
    pre_transform=T.Compose(
        [
            T.Center(),
            T.NormalizeRotation(),
            T.RadiusGraph(10),
            T.ToUndirected(),
            RandomWalkPE(20, "pe"),
        ]
    ),
    num_workers=4,
    chunk_size=256,
)
ds = FoldSeekDatasetSmall(
    pre_transform=T.Compose(
        [
            T.Center(),
            T.NormalizeRotation(),
            T.RadiusGraph(10),
            T.ToUndirected(),
            RandomWalkPE(20, "pe"),
        ]
    ),
)
print(len(ds))

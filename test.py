import os
import shutil
import sys

import pytorch_lightning as pl
import torch_geometric.transforms as T

from step.data import FoldSeekDataset, FoldSeekDatasetSmall, RandomWalkPE
from step.models import DenoiseModel

if os.path.exists("data/foldseek_small/processed"):
    shutil.rmtree("data/foldseek_small/processed")

ds = FoldSeekDataset(
    pre_transform=T.Compose(
        [
            T.Center(),
            T.NormalizeRotation(),
            T.RadiusGraph(10),
            T.ToUndirected(),
            RandomWalkPE(20, "pe"),
        ]
    ),
    num_workers=14,
    chunk_size=1000,
)

print(len(ds))

# model = DenoiseModel(hidden_dim=16, pe_dim=4, pos_dim=4, num_layers=4, heads=4)

# trainer = pl.Trainer(accelerator="cpu", devices=4, strategy="auto", logger=False)
# trainer.fit(model, datamodule=dm)

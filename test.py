import shutil
import sys

import pytorch_lightning as pl
import torch_geometric.transforms as T

from step.data import FoldSeekDatasetSmall, RandomWalkPE
from step.models import DenoiseModel

shutil.rmtree("data/foldseek_small/processed")

ds = FoldSeekDatasetSmall(
    pre_transform=T.Compose(
        [
            T.Center(),
            T.NormalizeRotation(),
            T.RadiusGraph(10),
            T.ToUndirected(),
            RandomWalkPE(20, "pe", cuda=True),
        ]
    ),
    num_workers=int(sys.argv[1]),
    chunk_size=1000,
)

print(len(ds))

# model = DenoiseModel(hidden_dim=16, pe_dim=4, pos_dim=4, num_layers=4, heads=4)

# trainer = pl.Trainer(accelerator="cpu", devices=4, strategy="auto", logger=False)
# trainer.fit(model, datamodule=dm)

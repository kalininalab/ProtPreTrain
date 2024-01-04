import os
import shutil
import sys

import torch_geometric.transforms as T

from step.data import FoldCompDataset, RandomWalkPE, ToCpu, ToCuda

ds_name = "afdb_rep_v4"
ds_folder = f"data/{ds_name}"
if os.path.exists(f"{ds_folder}/processed"):
    shutil.rmtree(f"{ds_folder}/processed")

ds = FoldCompDataset(
    db_name="afdb_rep_v4",
    pre_transform=T.Compose(
        [
            ToCuda(),
            T.Center(),
            T.NormalizeRotation(),
            T.RadiusGraph(10),
            RandomWalkPE(20, "pe", cuda=False),
            ToCpu(),
        ]
    ),
    num_workers=int(sys.argv[1]),
    chunk_size=int(sys.argv[2]),
)

print(len(ds))

# model = DenoiseModel(hidden_dim=16, pe_dim=4, pos_dim=4, num_layers=4, heads=4)

# trainer = pl.Trainer(accelerator="cpu", devices=4, strategy="auto", logger=False)
# trainer.fit(model, datamodule=dm)

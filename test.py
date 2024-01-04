import os
import shutil
import sys

import torch_geometric.transforms as T

from step.data import FoldCompDataset, RandomWalkPE, ToCpu, ToCuda
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_workers", type=int, default=1)
parser.add_argument("--chunk_size", type=int, default=1000)
args = parser.parse_args()

ds_name = "afdb_rep_v4"
ds_folder = f"data/{ds_name}"
if os.path.exists(f"{ds_folder}/processed"):
    shutil.rmtree(f"{ds_folder}/processed")

# wandb.init(project="data_processing", entity="rindti", config=args)

ds = FoldCompDataset(
    db_name="afdb_rep_v4",
    pre_transform=T.Compose(
        [
            # ToCuda(.8),
            T.Center(),
            T.NormalizeRotation(),
            T.RadiusGraph(10),
            RandomWalkPE(20, "pe", cuda=True),
            # ToCpu(),
        ]
    ),
    num_workers=args.num_workers,
    chunk_size=args.chunk_size,
)

print(len(ds))

# model = DenoiseModel(hidden_dim=16, pe_dim=4, pos_dim=4, num_layers=4, heads=4)

# trainer = pl.Trainer(accelerator="cpu", devices=4, strategy="auto", logger=False)
# trainer.fit(model, datamodule=dm)

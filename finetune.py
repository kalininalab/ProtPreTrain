import torch
import torch_geometric.transforms as T

import wandb
from step.data import FluorescenceDataModule

torch.set_float32_matmul_precision("medium")

# - class_path: torch_geometric.transforms.RadiusGraph
#         init_args:
#           r: 7.0
#       - class_path: torch_geometric.transforms.ToUndirected
#       - class_path: torch_geometric.transforms.Spherical

transforms = [T.RadiusGraph(7), T.ToUndirected(), T.Spherical()]
wandb.init(project="fluorescence", name="debug")
dm = FluorescenceDataModule("ilsenatorov/step/model-2lw34cxd:best", transforms=transforms)
dm.setup()

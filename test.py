import pytorch_lightning as pl
import torch_geometric.transforms as T

from step.data import FoldSeekDataModuleSmall, MaskType, PosNoise, RandomWalkPE
from step.models import DenoiseModel

dm = FoldSeekDataModuleSmall(
    pre_transforms=[
        T.Center(),
        T.NormalizeRotation(),
        T.RadiusGraph(10),
        T.ToUndirected(),
        RandomWalkPE(20, "pe"),
    ],
    transforms=[MaskType(0.15), PosNoise(1.0)],
    num_workers=1,
    batch_sampling=False,
    batch_size=2,
    # max_num_nodes=2048,
)
dm.setup()

dl = dm.train_dataloader()
print(dm.train[0])
batch = next(iter(dl))

model = DenoiseModel(hidden_dim=16, pe_dim=4, pos_dim=4, num_layers=4, heads=4)

trainer = pl.Trainer(accelerator="cpu", devices=1, strategy="auto")
trainer.fit(model, datamodule=dm)

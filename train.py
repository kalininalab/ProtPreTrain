from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch_geometric as pyg

import wandb
from step.data import FoldSeekDataModule, MaskType, MaskTypeAnkh, MaskTypeBERT, PosNoise, RandomWalkPE
from step.models import DenoiseModel

torch.set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy("file_system")
pl.seed_everything(42)

parser = ArgumentParser()
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--pe_dim", type=int, default=64)
parser.add_argument("--pos_dim", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--attn_type", type=str, default="performer")
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--predict_all", type=bool, default=True)
parser.add_argument("--posnoise", type=float, default=1.0)
parser.add_argument("--masktype", type=str, default="normal", choices=["normal", "ankh", "bert"])
parser.add_argument("--maskfrac", type=float, default=0.15)
parser.add_argument("--radius", type=int, default=7)
parser.add_argument("--walk_length", type=int, default=20)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--batch_sampling", type=bool, default=False)
parser.add_argument("--max_num_nodes", type=int, default=4096)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-4)


args = parser.parse_args()
config = vars(args)

model = DenoiseModel(**config)
masktype_transform = {"normal": MaskType, "ankh": MaskTypeAnkh, "bert": MaskTypeBERT}
datamodule = FoldSeekDataModule(
    transforms=[
        PosNoise(args.posnoise),
        masktype_transform[args.masktype](args.maskfrac),
        pyg.transforms.RadiusGraph(args.radius),
        pyg.transforms.ToUndirected(),
        pyg.transforms.AddRandomWalkPE(args.walk_length, name="pe"),
    ],
    pre_transforms=[pyg.transforms.Center(), pyg.transforms.NormalizeRotation()],
    batch_sampling=args.batch_sampling,
    batch_size=args.batch_size,
    max_num_nodes=args.max_num_nodes,
    num_workers=args.num_workers,
)
datamodule.setup()
logger = pl.loggers.WandbLogger(
    # offline=True,
    project="step",
    entity="rindti",
    # settings=wandb.Settings(start_method="fork"),
    config=config,
)
trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=args.max_epochs,
    precision="bf16-mixed",
    # strategy="auto",
    devices=-1,
    # num_nodes=args.num_nodes,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor="train/loss",
            mode="min",
            dirpath="checkpoints",
            save_last=True,
            save_on_train_epoch_end=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.RichProgressBar(),
        pl.callbacks.RichModelSummary(),
    ],
    logger=logger,
)
trainer.fit(model, datamodule=datamodule)

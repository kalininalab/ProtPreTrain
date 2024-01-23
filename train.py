from argparse import ArgumentParser

from step.utils import str_to_bool

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="afdb_rep_v4")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--pe_dim", type=int, default=64)
parser.add_argument("--pos_dim", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--attn_type", type=str, default="performer")
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--predict_all", type=str_to_bool, default=True)
parser.add_argument("--posnoise", type=float, default=1.0)
parser.add_argument("--masktype", type=str, default="normal", choices=["normal", "ankh", "bert"])
parser.add_argument("--maskfrac", type=float, default=0.15)
parser.add_argument("--radius", type=int, default=10)
parser.add_argument("--walk_length", type=int, default=20)
parser.add_argument("--batch_sampling", type=str_to_bool, default=False)
parser.add_argument("--max_num_nodes", type=int, default=4096, help="Max num nodes in a dynamic batch")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--subset", type=int, default=None)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--num_nodes", type=int, default=1, help="Computing nodes")
parser.add_argument("--num_workers", type=int, default=16)

args = parser.parse_args()

import pytorch_lightning as pl
import torch
import torch_geometric as pyg

import wandb
from step.data import FoldCompDataModule, MaskType, MaskTypeAnkh, MaskTypeBERT, PosNoise, RandomWalkPE
from step.models import DenoiseModel
from step.utils import WandbArtifactModelCheckpoint

torch.set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy("file_system")
pl.seed_everything(42)
config = vars(args)
logger = pl.loggers.WandbLogger(
    project="step",
    entity="rindti",
    settings=wandb.Settings(start_method="fork"),
    config=config,
    log_model=False,
)
masktype_transform = {"normal": MaskType, "ankh": MaskTypeAnkh, "bert": MaskTypeBERT}

datamodule = FoldCompDataModule(
    db_name=args.dataset,
    pre_transforms=[
        pyg.transforms.Center(),
        pyg.transforms.NormalizeRotation(),
        pyg.transforms.RadiusGraph(args.radius),
        pyg.transforms.ToUndirected(),
        RandomWalkPE(args.walk_length, attr_name="pe", cuda=True),
    ],
    transforms=[PosNoise(args.posnoise), masktype_transform[args.masktype](args.maskfrac)],
    batch_sampling=args.batch_sampling,
    batch_size=args.batch_size,
    max_num_nodes=args.max_num_nodes,
    num_workers=args.num_workers,
    subset=args.subset,
)
datamodule.setup()

run = logger.experiment
model = DenoiseModel(**config)
trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=args.max_epochs,
    precision="bf16-mixed",
    strategy="auto",
    devices=-1,
    num_nodes=args.num_nodes,
    callbacks=[
        WandbArtifactModelCheckpoint(
            wandb_run=run,
            monitor="train/loss",
            mode="min",
            dirpath=f"checkpoints/{run.id}",
            save_on_train_epoch_end=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.RichProgressBar(),
        pl.callbacks.RichModelSummary(),
    ],
    logger=logger,
    # profiler="pytorch"
)
trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

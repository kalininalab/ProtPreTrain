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
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--predict_all", type=str_to_bool, default=False)
parser.add_argument("--posnoise", type=float, default=1.0)
parser.add_argument("--masktype", type=str, default="normal", choices=["normal", "ankh", "bert"])
parser.add_argument("--maskfrac", type=float, default=0.15)
parser.add_argument("--radius", type=int, default=10)
parser.add_argument(
    "--pe",
    type=str,
    default="seq",
    choices=["rw", "seq", "none"],
    help="Positional encoding: random-walk (CPU-heavy, O(N^3)), sinusoidal residue index, or none",
)
parser.add_argument("--walk_length", type=int, default=20, help="Random-walk PE length (--pe rw only)")
parser.add_argument("--edge_dim", type=int, default=16, help="RBF edge-distance features; 0 disables edge features")
parser.add_argument(
    "--invariant",
    type=str_to_bool,
    default=False,
    help="Rotation-consistent model: no raw-coordinate input, noise predicted along neighbour directions",
)
parser.add_argument(
    "--clean_graph",
    type=str_to_bool,
    default=False,
    help="Ablation: build radius graph + PE from clean coordinates at process time (leaks noise-free structure)",
)
parser.add_argument(
    "--sequence_only",
    type=str_to_bool,
    default=False,
    help="Sequence-only control: replace coordinates with a straight line before noising and graph building",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_sampling", type=str_to_bool, default=False)
parser.add_argument("--max_num_nodes", type=int, default=4096, help="Max num nodes in a dynamic batch")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--max_length", type=int, default=1022, help="Drop structures longer than this (as in ESM)")
parser.add_argument("--subset", type=int, default=None)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument(
    "--scheduler",
    type=str,
    default="cosine",
    choices=["cosine", "legacy"],
    help="cosine: warmup + one cosine decay over the whole run; legacy: fixed 10k-step warmup, 100k-step cycle",
)
parser.add_argument("--warmup_frac", type=float, default=0.05, help="Fraction of total steps spent warming up")
parser.add_argument("--val_size", type=int, default=0, help="Structures held out for validation losses")
parser.add_argument("--num_nodes", type=int, default=1, help="Computing nodes")
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--wandb_project", type=str, default="step")
parser.add_argument("--summary_json", type=str, default=None, help="Write final metrics and checkpoint path here")

args = parser.parse_args()
if args.sequence_only and args.clean_graph:
    parser.error("--sequence_only needs the graph built at load time, so it can't be combined with --clean_graph")

import json
import time

import pytorch_lightning as pl
import torch
import torch_geometric as pyg
import wandb
from lightning.pytorch.strategies import DDPStrategy

from step.data import FoldCompDataModule, MaskType, MaskTypeAnkh, MaskTypeBERT, PosNoise
from step.data.transforms import SequenceOnly, graph_transforms
from step.models import DenoiseModel
from step.utils import WandbArtifactModelCheckpoint

# Explicitly specify the process group backend if you choose to

torch.set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy("file_system")
pl.seed_everything(args.seed)
config = vars(args)
logger = pl.loggers.WandbLogger(
    project=args.wandb_project,
    entity="rindti",
    settings=wandb.Settings(start_method="fork"),
    config=config,
    log_model=False,
)
masktype_transform = {"normal": MaskType, "ankh": MaskTypeAnkh, "bert": MaskTypeBERT}

# Graph + PE are built after PosNoise by default, so connectivity carries no information about the noise target.
# At load time RandomWalkPE stays on CPU: CUDA can't be initialised in forked dataloader workers.
graph = graph_transforms(args.radius, args.pe, args.walk_length, cuda=args.clean_graph)
pre_transforms = [pyg.transforms.Center(), pyg.transforms.NormalizeRotation()]
transforms = [PosNoise(args.posnoise), masktype_transform[args.masktype](args.maskfrac)]
if args.sequence_only:
    # Load-time rather than pre_transform, so existing processed chunks don't need reprocessing
    transforms.insert(0, SequenceOnly())
if args.clean_graph:
    pre_transforms += graph
else:
    transforms += graph

datamodule = FoldCompDataModule(
    db_name=args.dataset,
    pre_transforms=pre_transforms,
    transforms=transforms,
    batch_sampling=args.batch_sampling,
    batch_size=args.batch_size,
    max_num_nodes=args.max_num_nodes,
    num_workers=args.num_workers,
    subset=args.subset,
    max_length=args.max_length,
    val_size=args.val_size,
)

run = logger.experiment
model = DenoiseModel(**config)
checkpoint = WandbArtifactModelCheckpoint(
    wandb_run=run,
    monitor="train/loss",
    mode="min",
    dirpath=f"checkpoints/{run.id}",
    save_on_train_epoch_end=True,
)
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=args.max_epochs,
    # CPUs without native bf16 run bf16 autocast several times slower than fp32
    precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
    strategy="auto",
    devices="auto",
    # the dynamic batch sampler shards across ranks itself
    use_distributed_sampler=not args.batch_sampling,
    num_nodes=args.num_nodes,
    callbacks=[
        checkpoint,
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.RichProgressBar(),
        pl.callbacks.RichModelSummary(),
    ],
    logger=logger,
    limit_val_batches=1.0 if args.val_size else 0,
    # profiler="pytorch"
)
start = time.time()
trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
train_time = time.time() - start
# Pretraining compute for reporting; params are counted after fit because the model has lazy layers
if trainer.is_global_zero:
    run.summary["num_params"] = sum(p.numel() for p in model.parameters())
    run.summary["world_size"] = trainer.world_size
    run.summary["train_time_s"] = train_time
    if args.summary_json:
        summary = {k: v.item() for k, v in trainer.callback_metrics.items()}
        summary.update(
            train_time_s=train_time,
            num_params=run.summary["num_params"],
            ckpt_path=checkpoint.best_model_path,
            run_id=run.id,
        )
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)

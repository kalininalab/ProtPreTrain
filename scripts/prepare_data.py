"""Download and process a foldcomp database for pretraining, and cache its residue counts.

Runs the same preparation train.py does on first use (identical pre-transforms, via FoldCompDataModule), but as its
own CPU job, so GPU jobs never sit idle while a database downloads or processes:

    python scripts/prepare_data.py --dataset afdb_rep_v4 --num_workers 8
"""

import argparse

import numpy as np
import torch_geometric.transforms as T

from step.data import FoldCompDataModule

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="foldcomp database name, e.g. afdb_rep_v4 or e_coli")
parser.add_argument("--num_workers", type=int, default=8)
args = parser.parse_args()

# Must match train.py's pre_transforms: they are baked into the processed chunks
dm = FoldCompDataModule(
    db_name=args.dataset,
    pre_transforms=[T.Center(), T.NormalizeRotation()],
    num_workers=args.num_workers,
    max_length=1022,  # any value: makes prepare_data() also cache lengths.npy
)
dm.prepare_data()
ds = dm._dataset()
lengths = ds.lengths()
print(
    f"{args.dataset}: {len(ds)} structures, median {int(np.median(lengths))} residues, {(lengths <= 1022).sum()} <= 1022"
)

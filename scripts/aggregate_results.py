"""Aggregate downstream test metrics over seeds into mean ± 95% CI, and pretraining compute per model.

python scripts/aggregate_results.py --out results.csv
"""

import argparse

import pandas as pd
import wandb
from scipy import stats

GROUP_KEYS = ["dataset", "model_source", "model", "ablation", "ablation_maskfrac", "random_init"]

parser = argparse.ArgumentParser()
parser.add_argument("--entity", default="rindti")
parser.add_argument("--projects", nargs="+", default=["fluorescence", "stability", "homology", "dti"])
parser.add_argument("--out", default=None, help="Optional CSV output path")
args = parser.parse_args()

api = wandb.Api()


def ci95(x: pd.Series) -> float:
    """Half-width of the 95% t-interval of the mean (NaN for a single run)."""
    return stats.sem(x) * stats.t.ppf(0.975, len(x) - 1) if len(x) > 1 else float("nan")


rows = []
for project in args.projects:
    for run in api.runs(f"{args.entity}/{project}", filters={"state": "finished"}):
        row = {k: run.config.get(k) for k in GROUP_KEYS}
        row["dataset"] = project
        row.update({k: v for k, v in run.summary.items() if k.startswith("test") and isinstance(v, (int, float))})
        rows.append(row)

df = pd.DataFrame(rows)
# Old runs predate some flags; fill with the defaults so they group with explicit-default runs
df = df.fillna({"ablation": "none", "ablation_maskfrac": 1.0, "random_init": False})
metrics = [c for c in df.columns if c.startswith("test")]
res = df.groupby(GROUP_KEYS, dropna=False)[metrics].agg(["mean", ci95, "count"])
res = res.dropna(axis=1, how="all")
pd.set_option("display.width", 200)
print(res.round(4).to_string())

pre = pd.DataFrame(
    [
        {
            "run": run.id,
            "num_params": run.summary.get("num_params"),
            "gpu_hours": run.summary.get("_runtime", 0) * run.summary.get("world_size", 4) / 3600,
        }
        for run in api.runs(f"{args.entity}/step", filters={"state": "finished"})
    ]
)
print("\nPretraining compute:")
print(pre.round(2).to_string(index=False))

if args.out:
    res.to_csv(args.out)

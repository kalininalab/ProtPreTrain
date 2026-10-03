"""Aggregate downstream test metrics over seeds into mean ± 95% CI, and pretraining compute per model.

Reads finished runs from the MLflow tracking store ($MLFLOW_TRACKING_URI, default sqlite:///mlflow.db).

python scripts/aggregate_results.py --out results.csv
"""

import argparse

import mlflow
import pandas as pd
from scipy import stats

from step.utils import tracking_uri

GROUP_KEYS = ["dataset", "model_source", "model", "ablation", "ablation_maskfrac", "random_init"]
# MLflow params are strings; runs that predate a flag get its default so they group with explicit-default runs
PARAM_DEFAULTS = {"ablation": "none", "ablation_maskfrac": "1.0", "random_init": "False"}

parser = argparse.ArgumentParser()
parser.add_argument("--tracking_uri", default=None, help="Defaults to $MLFLOW_TRACKING_URI or sqlite:///mlflow.db")
parser.add_argument("--experiments", nargs="+", default=["fluorescence", "stability", "homology", "dti"])
parser.add_argument("--pretrain_experiment", default="step")
parser.add_argument("--out", default=None, help="Optional CSV output path")
args = parser.parse_args()

mlflow.set_tracking_uri(args.tracking_uri or tracking_uri())


def finished_runs(experiments: list) -> pd.DataFrame:
    """Finished runs of the given experiments, one row each (params.* and metrics.* columns)."""
    return mlflow.search_runs(
        experiment_names=experiments, filter_string="attributes.status = 'FINISHED'", output_format="pandas"
    )


def metric(runs: pd.DataFrame, name: str) -> pd.Series:
    """A metric column of search_runs output, NaN where (or if) runs never logged it."""
    return runs.get(f"metrics.{name}", pd.Series(float("nan"), index=runs.index))


def ci95(x: pd.Series) -> float:
    """Half-width of the 95% t-interval of the mean (NaN for a single run)."""
    return stats.sem(x) * stats.t.ppf(0.975, len(x) - 1) if len(x) > 1 else float("nan")


runs = finished_runs(args.experiments)
if runs.empty:
    raise SystemExit(f"No finished runs in experiments {args.experiments}")
df = pd.DataFrame({k: runs.get(f"params.{k}") for k in GROUP_KEYS})
df = df.fillna(PARAM_DEFAULTS)
metric_cols = [c for c in runs.columns if c.startswith("metrics.test")]
for c in metric_cols:
    df[c.removeprefix("metrics.")] = runs[c]
metrics = [c.removeprefix("metrics.") for c in metric_cols]
res = df.groupby(GROUP_KEYS, dropna=False)[metrics].agg(["mean", ci95, "count"])
res = res.dropna(axis=1, how="all")
pd.set_option("display.width", 200)
print(res.round(4).to_string())

pre = finished_runs([args.pretrain_experiment])
if not pre.empty:
    compute = pd.DataFrame(
        {
            "run": pre["run_id"],
            "num_params": metric(pre, "num_params"),
            "gpu_hours": metric(pre, "train_time_s") * metric(pre, "world_size") / 3600,
        }
    )
    print("\nPretraining compute:")
    print(compute.round(2).to_string(index=False))

if args.out:
    res.to_csv(args.out)

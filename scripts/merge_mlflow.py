"""Merge many MLflow SQLite stores into one, e.g. to browse cluster runs (one store per job) in a single ``mlflow ui``.

    rsync -a conduit:/home/s8ilsena/step/runs/ runs_conduit/
    python scripts/merge_mlflow.py --src 'runs_conduit/**/*.db' --dest sqlite:///merged.db
    mlflow ui --backend-store-uri sqlite:///merged.db

Copies experiments (by name), and every run's params, tags, full metric histories, status and timestamps. Re-running
is incremental: a run already merged is skipped unless its source has changed since (status or end time), in which
case its copy is replaced. Artifacts are not copied (the training scripts log none).
"""

import argparse
import glob
import os

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

SOURCE_TAG = "merge.source"  # "<source db path>:<source run id>" on every merged run
VERSION_TAG = "merge.source_version"  # "<status>:<end_time>" of the source run when it was copied
BATCH = 1000  # MLflow's log_batch limit for metrics (params: 100, tags: 100)


def chunks(items: list, n: int):
    """Consecutive slices of at most n items."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


def copy_run(src: MlflowClient, dst: MlflowClient, run, experiment_id: str, source: str, version: str) -> None:
    """Recreate ``run`` from ``src`` in ``dst`` under ``experiment_id``."""
    tags = {k: v for k, v in run.data.tags.items() if not k.startswith("merge.")}
    tags.update({SOURCE_TAG: source, VERSION_TAG: version})
    new = dst.create_run(experiment_id, start_time=run.info.start_time, tags=tags, run_name=run.info.run_name)
    rid = new.info.run_id
    for batch in chunks([Param(k, v) for k, v in run.data.params.items()], 100):
        dst.log_batch(rid, params=batch)
    history = [
        Metric(m.key, m.value, m.timestamp, m.step)
        for key in run.data.metrics
        for m in src.get_metric_history(run.info.run_id, key)
    ]
    for batch in chunks(history, BATCH):
        dst.log_batch(rid, metrics=batch)
    dst.set_terminated(rid, status=run.info.status, end_time=run.info.end_time)


def merge_store(path: str, dst: MlflowClient) -> tuple[int, int]:
    """Merge one source store into ``dst``; returns (copied, skipped) run counts."""
    src = MlflowClient(f"sqlite:///{os.path.abspath(path)}")
    copied = skipped = 0
    for exp in src.search_experiments():
        target = dst.get_experiment_by_name(exp.name)
        exp_id = target.experiment_id if target else dst.create_experiment(exp.name)
        for run in src.search_runs([exp.experiment_id], max_results=50000):
            source = f"{os.path.abspath(path)}:{run.info.run_id}"
            version = f"{run.info.status}:{run.info.end_time}"
            existing = dst.search_runs([exp_id], filter_string=f"tags.`{SOURCE_TAG}` = '{source}'")
            if existing and existing[0].data.tags.get(VERSION_TAG) == version:
                skipped += 1
                continue
            for old in existing:
                dst.delete_run(old.info.run_id)
            copy_run(src, dst, run, exp_id, source, version)
            copied += 1
    return copied, skipped


def main():
    """Parse arguments and merge every matching store."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Recursive glob of source SQLite store files")
    parser.add_argument("--dest", default="sqlite:///merged.db", help="Destination tracking URI")
    args = parser.parse_args()
    paths = [p for p in sorted(glob.glob(args.src, recursive=True)) if f"sqlite:///{os.path.abspath(p)}" != args.dest]
    if not paths:
        raise SystemExit(f"No stores match {args.src}")
    dst = MlflowClient(args.dest)
    total_copied = total_skipped = 0
    for path in paths:
        copied, skipped = merge_store(path, dst)
        total_copied += copied
        total_skipped += skipped
        print(f"{path}: {copied} copied, {skipped} unchanged")
    print(f"{len(paths)} stores -> {args.dest}: {total_copied} runs copied, {total_skipped} unchanged")


if __name__ == "__main__":
    main()

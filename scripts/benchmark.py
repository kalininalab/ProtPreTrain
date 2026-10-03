"""Small-scale benchmark harness for STEP pretraining changes.

Subcommands (run from the repo root):

    python scripts/benchmark.py loader                 # per-sample load-time transform cost, pe=rw vs seq/none
    python scripts/benchmark.py pretrain --seeds 0 1 2 # CONFIGS x seeds through train.py (resumable)
    python scripts/benchmark.py probe --head_seeds 0 1 2  # frozen-embedding homology probe of every finished run
    python scripts/benchmark.py table                  # bench/results.csv + bench/results.md

Every run lives in ``<out>/<config>_s<seed>/`` (``pretrain.json``, ``pretrain.log``, ``probe_h<head_seed>.json``,
``probe_h<head_seed>.log``). A run whose json already exists is skipped, so any subcommand can be re-run after an
interruption. Subprocesses log to MLflow in ``<out>/mlflow.db`` (experiment ``step-bench``), separate from the
main tracking store.

On the conduit cluster, ``--condor FILE`` (pretrain and probe) writes an HTCondor queue file for hpc/gpu.sub instead
of running anything: one ``<mlflow store>, <command>`` line per job, each job with its own MLflow store. Generate the
probe file once the pretraining jobs have finished (it reads their ``pretrain.json``). See hpc/README.md.

Unknown arguments are passed through verbatim to train.py / finetune.py after the preset/config arguments (argparse
keeps the last occurrence, so they override), e.g. ``pretrain --configs legacy -- --max_epochs 2 --num_workers 4``.
``--preset scale`` swaps the common arguments for the cluster-sized setup (AFDB subset, bigger model).
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------------------------------------------------
# Experiment definition — edit here.
# ---------------------------------------------------------------------------------------------------------------------

PRESETS = {
    # CPU/laptop-sized: E. coli proteome (symlinked foldcomp DB in data/e_coli_bench/raw)
    "local": {
        "dataset": "e_coli_bench",
        "subset": 4400,
        "hidden_dim": 128,
        "pe_dim": 32,
        "pos_dim": 32,
        "num_layers": 4,
        "batch_size": 16,
        "val_size": 400,
        "max_length": 512,
        "max_epochs": 2,
        "num_workers": 5,
    },
    # Cluster-sized: not run locally
    "scale": {
        "dataset": "afdb_rep_v4",
        "subset": 500_000,
        "hidden_dim": 512,
        "pe_dim": 64,
        "pos_dim": 64,
        "num_layers": 12,
        "batch_size": 32,
        "val_size": 5000,
        "max_length": 1022,
        "max_epochs": 5,
        "num_workers": 16,
    },
}

LEGACY = {
    "pe": "rw",
    "dropout": 0.5,
    "scheduler": "legacy",
    "lr": 1e-4,
    "edge_dim": 0,
    "invariant": "false",
}


def _cumulative(base: dict, steps: list) -> list:
    """Build a cumulative ablation: each (name, overrides) step applies on top of the previous config."""
    configs, current = [], dict(base)
    for name, overrides in steps:
        current = {**current, **overrides}
        configs.append({"name": name, **current})
    return configs


CONFIGS = _cumulative(
    LEGACY,
    [
        ("legacy", {}),
        ("+sched", {"scheduler": "cosine"}),
        ("+lr3e-4", {"lr": 3e-4}),
        ("+drop0.1", {"dropout": 0.1}),
        ("+seqpe", {"pe": "seq"}),
        ("+edges", {"edge_dim": 16}),
        ("+invariant", {"invariant": "true"}),
    ],
)
_EDGES = next(c for c in CONFIGS if c["name"] == "+edges")
CONFIGS += [
    {**_EDGES, "name": "drop0", "dropout": 0.0},
    {**_EDGES, "name": "nope", "pe": "none"},
    {**_EDGES, "name": "lr1e-3", "lr": 1e-3},
]
# The random-init probe control reuses this config's checkpoints (architecture + hparams, fresh weights)
RANDOM_INIT_OF = "+invariant"  # the final cumulative config

PROBE = {"dataset": "homology", "max_epochs": 200}
PRETRAIN_METRICS = ["train_time_s", "num_params", "val/pred_acc", "val/noise_loss", "val/pred_loss", "train/loss"]
PROBE_METRICS = ["test_fold/acc", "test_superfamily/acc", "test_family/acc"]
EXPERIMENT = "step-bench"


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------


def to_cli(d: dict) -> list:
    """Turn a dict into ``--key value`` arguments, skipping the ``name`` key and None values."""
    out = []
    for k, v in d.items():
        if k == "name" or v is None:
            continue
        out += [f"--{k}", str(v)]
    return out


def run_dir(out: Path, name: str, seed: int) -> Path:
    """Directory holding everything for one (config, pretraining seed) pair."""
    return out / f"{name}_s{seed}"


def run_cmd(cmd: list, log: Path, dry_run: bool, tracking_db: Path) -> int:
    """Run a subprocess in the repo root, logging to MLflow in ``tracking_db`` and its output to ``log``; returns the exit code."""
    print(("[dry-run] " if dry_run else "") + " ".join(cmd) + f"  > {log}", flush=True)
    if dry_run:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{tracking_db}"}
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  -> exit {rc} after {time.time() - t0:.0f}s ({log})", flush=True)
    return rc


def write_condor(path: str, lines: list) -> None:
    """Write ``<store>, <command>`` lines for hpc/gpu.sub; condor splits on commas, so commands must not contain any."""
    for _store, cmd in lines:
        if "," in " ".join(cmd):
            sys.exit(f"command contains a comma, which condor would split on: {' '.join(cmd)}")
    Path(path).write_text("".join(f"{store}, {' '.join(cmd)}\n" for store, cmd in lines))
    print(f"wrote {len(lines)} jobs to {path}; submit with: condor_submit -a 'runfile={path}' hpc/gpu.sub")


def run_all(jobs: list, n_parallel: int) -> list:
    """Run a list of zero-argument callables, optionally in parallel threads (each launches a subprocess)."""
    if n_parallel <= 1:
        return [j() for j in jobs]
    with ThreadPoolExecutor(n_parallel) as ex:
        return list(ex.map(lambda j: j(), jobs))


def select_configs(names: list) -> list:
    """Configs whose name is in ``names`` (all when empty)."""
    if not names:
        return CONFIGS
    known = {c["name"] for c in CONFIGS}
    missing = set(names) - known
    if missing:
        sys.exit(f"Unknown configs {sorted(missing)}; known: {sorted(known)}")
    return [c for c in CONFIGS if c["name"] in names]


# ---------------------------------------------------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------------------------------------------------


def cmd_loader(args, _passthrough):
    """Time PosNoise + MaskType + graph transforms per sample on raw FoldCompDataset items, for each PE type."""
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import numpy as np
    import torch
    import torch_geometric.transforms as T

    from step.data.datasets import FoldCompDataset
    from step.data.transforms import MaskType, PosNoise

    try:
        from step.data.transforms import graph_transforms
    except ImportError:  # older checkout: rebuild the equivalent list

        def graph_transforms(radius, pe="rw", walk_length=20, **_):
            from step.data.transforms import RandomWalkPE

            out = [T.RadiusGraph(radius), T.ToUndirected()]
            return out + ([RandomWalkPE(walk_length, attr_name="pe", cuda=False)] if pe == "rw" else [])

    torch.set_num_threads(1)  # one dataloader worker = one thread
    ds = FoldCompDataset(
        db_name=args.dataset,
        pre_transform=T.Compose([T.Center(), T.NormalizeRotation()]),
        num_workers=args.num_workers,
    )
    lengths = ds.lengths()
    eligible = np.flatnonzero(lengths <= args.max_length)
    rng = np.random.default_rng(0)
    idx = rng.choice(eligible, size=min(args.n, len(eligible)), replace=False)
    print(f"{args.dataset}: {len(ds)} structures, {len(eligible)} <= {args.max_length} residues; timing {len(idx)}")

    t0 = time.perf_counter()
    samples = [ds.get(int(i)) for i in idx]
    read_ms = (time.perf_counter() - t0) / len(samples) * 1e3
    lens = np.array([s.num_nodes for s in samples])

    bins = [0, 128, 256, 512, 768, args.max_length + 1]
    results = {"dataset": args.dataset, "n": len(samples), "read_ms_mean": read_ms, "pe": {}}
    for pe in args.pe:
        transform = T.Compose(
            [PosNoise(args.posnoise), MaskType(args.maskfrac)] + graph_transforms(args.radius, pe, args.walk_length)
        )
        for s in samples[:3]:  # warm-up
            transform(s.clone())
        times = []
        for s in samples:
            s = s.clone()
            t0 = time.perf_counter()
            transform(s)
            times.append((time.perf_counter() - t0) * 1e3)
        times = np.array(times)
        per_bin = {}
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            sel = (lens >= lo) & (lens < hi)
            if sel.any():
                per_bin[f"{lo}-{hi - 1}"] = {"n": int(sel.sum()), "ms": float(times[sel].mean())}
        total = times.mean() + read_ms
        results["pe"][pe] = {
            "transform_ms_mean": float(times.mean()),
            "transform_ms_p95": float(np.percentile(times, 95)),
            "samples_per_s_per_worker": float(1e3 / total),
            "by_length": per_bin,
        }

    print(f"\nHDF5 read: {read_ms:.2f} ms/sample (single thread)")
    print(f"{'pe':>6} {'transform ms':>13} {'p95 ms':>8} {'samples/s/worker (incl. read)':>30}")
    for pe, r in results["pe"].items():
        print(
            f"{pe:>6} {r['transform_ms_mean']:>13.2f} {r['transform_ms_p95']:>8.2f} "
            f"{r['samples_per_s_per_worker']:>30.1f}"
        )
    print("\nms/sample by length bin:")
    header = sorted({b for r in results["pe"].values() for b in r["by_length"]}, key=lambda b: int(b.split("-")[0]))
    print(f"{'pe':>6} " + " ".join(f"{b:>10}" for b in header))
    for pe, r in results["pe"].items():
        print(f"{pe:>6} " + " ".join(f"{r['by_length'].get(b, {}).get('ms', float('nan')):>10.2f}" for b in header))
    print(
        "   n   "
        + " ".join(f"{next(iter(results['pe'].values()))['by_length'].get(b, {}).get('n', 0):>10}" for b in header)
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "loader.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out / 'loader.json'}")


def cmd_pretrain(args, passthrough):
    """Run every selected config x seed through train.py, skipping runs that already wrote their summary json."""
    out = Path(args.out).resolve()
    python = "python" if args.condor else sys.executable  # the image's python on the cluster
    jobs, condor = [], []
    for cfg, seed in itertools.product(select_configs(args.configs), args.seeds):
        d = run_dir(out, cfg["name"], seed)
        summary = d / "pretrain.json"
        if summary.exists():
            print(f"skip {d.name}: {summary.name} exists")
            continue
        cmd = (
            [python, "train.py"]
            + to_cli(PRESETS[args.preset])
            + to_cli(cfg)
            + ["--seed", str(seed), "--experiment", EXPERIMENT, "--summary_json", str(summary)]
            # restart-safe: an interrupted run continues from its last.ckpt when rerun
            + ["--ckpt_dir", str(d / "checkpoints"), "--resume", "auto"]
            + passthrough
        )
        if not args.dry_run:
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps({"preset": args.preset, **cfg, "seed": seed}, indent=2))
        condor.append((d / "pretrain.mlflow.db", cmd))
        jobs.append(lambda cmd=cmd, d=d: run_cmd(cmd, d / "pretrain.log", args.dry_run, out / "mlflow.db"))
    if args.condor:
        write_condor(args.condor, condor)
        return
    rcs = run_all(jobs, args.jobs)
    failed = sum(rc != 0 for rc in rcs)
    print(f"pretrain: {len(rcs)} launched, {failed} failed")


def _probe_targets(out: Path, configs: list) -> list:
    """(label, run dir, pretrain json, random_init) for every finished pretraining run, plus random-init controls."""
    targets = []
    for cfg in configs:
        for d in sorted(out.glob(f"{glob_escape(cfg['name'])}_s*")):
            if (d / "pretrain.json").exists():
                targets.append((d, d / "pretrain.json", False))
                if cfg["name"] == RANDOM_INIT_OF:
                    seed = d.name.rsplit("_s", 1)[1]
                    targets.append((run_dir(out, "random_init", int(seed)), d / "pretrain.json", True))
    return targets


def glob_escape(s: str) -> str:
    """Escape glob metacharacters in a config name."""
    return "".join(f"[{c}]" if c in "[]*?" else c for c in s)


def cmd_probe(args, passthrough):
    """Homology probe (frozen encoder + MLP head) for each finished pretraining run and the random-init control."""
    out = Path(args.out).resolve()
    python = "python" if args.condor else sys.executable
    jobs, condor = [], []
    for d, pretrain_json, random_init in _probe_targets(out, select_configs(args.configs)):
        ckpt = json.loads(pretrain_json.read_text()).get("ckpt_path")
        if not ckpt or not Path(ckpt).exists():
            print(f"skip {d.name}: checkpoint {ckpt!r} missing")
            continue
        for hs in args.head_seeds:
            summary = d / f"probe_h{hs}.json"
            if summary.exists():
                print(f"skip {d.name} head seed {hs}: exists")
                continue
            cmd = (
                [python, "finetune.py"]
                + to_cli({**PROBE, "max_epochs": args.max_epochs})
                + ["--model_source", "checkpoint", "--model", str(ckpt), "--seed", str(hs)]
                + ["--experiment", EXPERIMENT, "--summary_json", str(summary)]
                # the encoder is frozen, so every head seed reuses one set of embeddings per run
                + ["--embed_cache", str(d / "embeddings")]
                + (["--random_init"] if random_init else [])
                + passthrough
            )
            if not args.dry_run:
                d.mkdir(parents=True, exist_ok=True)
            condor.append((d / f"probe_h{hs}.mlflow.db", cmd))
            jobs.append(
                lambda cmd=cmd, d=d, hs=hs: run_cmd(cmd, d / f"probe_h{hs}.log", args.dry_run, out / "mlflow.db")
            )
    if not jobs:
        print("probe: nothing to do")
        return
    if args.condor:
        # Head seeds of one run share --embed_cache and data/homology; finetune.py file-locks both, so they can all
        # be queued at once (the first embeds, the rest wait and load)
        write_condor(args.condor, condor)
        return
    # The first probe processes data/homology for this graph setup; run it alone so parallel jobs don't race on it
    rcs = [jobs[0]()] + run_all(jobs[1:], args.jobs)
    print(f"probe: {len(rcs)} launched, {sum(rc != 0 for rc in rcs)} failed")


def cmd_table(args, _passthrough):
    """Collect all jsons into results.csv (one row per pretraining run) and results.md (mean ± std over seeds)."""
    import numpy as np
    import pandas as pd

    out = Path(args.out)
    order = [c["name"] for c in CONFIGS] + ["random_init"]
    rows = []
    for d in sorted(p for p in out.iterdir() if p.is_dir() and "_s" in p.name):
        name, seed = d.name.rsplit("_s", 1)
        row = {"config": name, "seed": int(seed)}
        pre = d / "pretrain.json"
        if name == "random_init":
            pre = run_dir(out, RANDOM_INIT_OF, int(seed)) / "pretrain.json"
            row["config"] = f"random_init ({RANDOM_INIT_OF})"
        if pre.exists():
            p = json.loads(pre.read_text())
            if name != "random_init":
                row.update({k: p.get(k) for k in PRETRAIN_METRICS})
            else:
                row["num_params"] = p.get("num_params")
        probes = [json.loads(f.read_text()) for f in sorted(d.glob("probe_h*.json"))]
        row["n_head_seeds"] = len(probes)
        for m in PROBE_METRICS:
            vals = [pr[m] for pr in probes if pr.get(m) is not None]
            row[m] = float(np.mean(vals)) if vals else None  # head seeds averaged within a pretraining seed
        rows.append(row)
    if not rows:
        sys.exit(f"no runs under {out}")
    df = pd.DataFrame(rows)
    rank = {n: i for i, n in enumerate(order)}
    df["_order"] = df["config"].map(lambda c: rank.get(c.split(" ")[0], len(order)))
    df = df.sort_values(["_order", "seed"]).drop(columns="_order")
    df.to_csv(out / "results.csv", index=False)

    cols = [
        ("train_time_s", "train time (min)", 1 / 60, 1),
        ("num_params", "params (M)", 1e-6, 2),
        ("val/pred_acc", "val pred acc", 1, 3),
        ("val/noise_loss", "val noise loss", 1, 3),
        ("test_fold/acc", "fold acc", 1, 3),
        ("test_superfamily/acc", "superfam acc", 1, 3),
        ("test_family/acc", "family acc", 1, 3),
    ]
    lines = ["| config | n | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for config, g in df.groupby("config", sort=False):
        cells = []
        for key, _, scale, digits in cols:
            v = pd.to_numeric(g.get(key), errors="coerce").dropna() * scale if key in g else pd.Series(dtype=float)
            if v.empty:
                cells.append("–")
            elif len(v) == 1:
                cells.append(f"{v.iloc[0]:.{digits}f}")
            else:
                cells.append(f"{v.mean():.{digits}f} ± {v.std():.{digits}f}")
        lines.append(f"| {config} | {len(g)} | " + " | ".join(cells) + " |")
    md = (
        "\n".join(lines)
        + "\n\nmean ± std over pretraining seeds; probe accuracies are first averaged over head seeds.\n"
    )
    (out / "results.md").write_text(md)
    print(md)
    print(f"wrote {out / 'results.csv'} and {out / 'results.md'}")


def main():
    """Parse arguments and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "bench"), help="Benchmark output directory")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("loader", help="Per-sample load-time transform cost")
    p.add_argument("--dataset", default="e_coli_bench")
    p.add_argument("--n", type=int, default=300, help="Number of structures to time")
    p.add_argument("--pe", nargs="+", default=["rw", "none"], help="'seq' and 'none' need no transform")
    p.add_argument("--max_length", type=int, default=1022)
    p.add_argument("--radius", type=int, default=10)
    p.add_argument("--walk_length", type=int, default=20)
    p.add_argument("--posnoise", type=float, default=1.0)
    p.add_argument("--maskfrac", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=10, help="Only used if the dataset still needs processing")

    for name, helptext in [("pretrain", "Run CONFIGS x seeds via train.py"), ("probe", "Homology probe per run")]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--configs", nargs="*", default=[], help="Subset of config names (default: all)")
        p.add_argument("--jobs", type=int, default=1, help="Concurrent subprocesses")
        p.add_argument("--dry_run", action="store_true", help="Print commands without running them")
        p.add_argument(
            "--condor", default=None, help="Write an HTCondor queue file for hpc/gpu.sub instead of running"
        )
    sub.choices["pretrain"].add_argument("--preset", choices=sorted(PRESETS), default="local")
    sub.choices["pretrain"].add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    sub.choices["probe"].add_argument("--head_seeds", type=int, nargs="+", default=[0, 1, 2])
    sub.choices["probe"].add_argument("--max_epochs", type=int, default=PROBE["max_epochs"])

    sub.add_parser("table", help="Aggregate jsons into results.csv / results.md")

    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if passthrough and args.command not in ("pretrain", "probe"):
        parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
    {"loader": cmd_loader, "pretrain": cmd_pretrain, "probe": cmd_probe, "table": cmd_table}[args.command](
        args, passthrough
    )


if __name__ == "__main__":
    main()

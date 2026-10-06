"""Small-scale benchmark harness for STEP pretraining changes.

Subcommands (run from the repo root):

    python scripts/benchmark.py loader                 # per-sample load-time transform cost, pe=rw vs seq/none
    python scripts/benchmark.py pretrain --seeds 0 1 2 # CONFIGS x seeds through train.py (resumable)
    python scripts/benchmark.py probe --head_seeds 0 1 2  # frozen-embedding homology probe of every finished run
    python scripts/benchmark.py probe --dataset stability  # same for another downstream dataset (PROBE_METRICS)
    python scripts/benchmark.py table                  # bench/results.csv + bench/results.md, every probed dataset
    python scripts/benchmark.py dynamics pretrain|probe|analyze  # probes of checkpoints kept during pretraining

Every run lives in ``<out>/<config>_s<seed>/`` (``pretrain.json``, ``pretrain.log``, ``checkpoints/``, and per head
seed ``probe_h<head_seed>.json`` + ``.log`` for homology, ``probe_<dataset>_h<head_seed>.json`` + ``.log`` for any
other dataset). Probe jobs of one run share frozen-encoder embeddings in ``embeddings/`` (homology) or
``embeddings_<dataset>/``. A run whose json already exists is skipped, so any subcommand can be re-run after an
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

PROBE = {"dataset": "homology", "max_epochs": 200}  # defaults of the probe subcommand
PRETRAIN_METRICS = ["train_time_s", "num_params", "val/pred_acc", "val/noise_loss", "val/pred_loss", "train/loss"]
# Probe datasets (finetune.py --dataset) -> {test metric in its summary json: results.md column label}
PROBE_METRICS = {
    "homology": {"test_fold/acc": "fold acc", "test_superfamily/acc": "superfam acc", "test_family/acc": "family acc"},
    "fluorescence": {"test/spearman": "fluorescence ρ"},
    "stability": {"test/spearman": "stability ρ"},
    "deeploc": {"test/acc": "deeploc acc"},  # 10-class subcellular localisation
}
EXPERIMENT = "step-bench"
MAX_CONDOR_JOBS = 150  # SUBMIT_REQUIREMENT_MaxMaterializations on conduit

# Pretraining-dynamics study (``dynamics`` subcommand): one config pretrained with checkpoints kept along the way,
# each kept checkpoint probed downstream. Lives in <out>/dynamics/<config>_s<seed>/.
DYNAMICS = {
    "config": "+invariant",
    "keep_steps": ["double=500", "final"],  # 500, 1k, 2k, ..., 64k and the last step (77,345 at the scale preset)
    "ckpt_every_n_steps": 2000,  # last.ckpt refresh: a preempted job loses at most this many steps
    "experiment": "step-dynamics",
}
# Pretraining quantities correlated with downstream metrics (keys of each kept checkpoint's sidecar json)
DYNAMICS_PRETRAIN = ["val/loss", "val/noise_loss", "val/pred_loss", "val/pred_acc", "train/loss_window"]


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


def write_condor(path: str, lines: list, max_jobs: int = MAX_CONDOR_JOBS) -> list:
    """Write ``<store>, <command>`` lines for hpc/gpu.sub; condor splits on commas, so commands must not contain any.

    More than ``max_jobs`` lines (the per-submit limit on conduit) go to ``<stem>_part<i><suffix>`` files, each
    submitted separately. Returns the paths written.
    """
    for _store, cmd in lines:
        if "," in " ".join(cmd):
            sys.exit(f"command contains a comma, which condor would split on: {' '.join(cmd)}")
    path = Path(path)
    chunks = [lines[i : i + max_jobs] for i in range(0, len(lines), max_jobs)] or [[]]
    if len(chunks) == 1:
        paths = [path]
    else:
        paths = [path.with_name(f"{path.stem}_part{i + 1}{path.suffix}") for i in range(len(chunks))]
    for p, chunk in zip(paths, chunks, strict=True):
        p.write_text("".join(f"{store}, {' '.join(cmd)}\n" for store, cmd in chunk))
        print(f"wrote {len(chunk)} jobs to {p}; submit with: condor_submit -a 'runfile={p}' hpc/gpu.sub")
    return paths


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


def probe_stem(dataset: str, head_seed) -> str:
    """File stem of one probe's json/log/mlflow store; homology keeps the original ``probe_h<seed>`` names."""
    return f"probe_h{head_seed}" if dataset == "homology" else f"probe_{dataset}_h{head_seed}"


def embed_cache(d: Path, dataset: str) -> Path:
    """Per-run, per-dataset embedding cache (finetune.py keys caches by split name only, so datasets can't share)."""
    return d / ("embeddings" if dataset == "homology" else f"embeddings_{dataset}")


def glob_escape(s: str) -> str:
    """Escape glob metacharacters in a config name."""
    return "".join(f"[{c}]" if c in "[]*?" else c for c in s)


def cmd_probe(args, passthrough):
    """Downstream probe (frozen encoder + MLP head) for each finished pretraining run and the random-init control."""
    out = Path(args.out).resolve()
    python = "python" if args.condor else sys.executable
    jobs, condor = [], []
    for d, pretrain_json, random_init in _probe_targets(out, select_configs(args.configs)):
        ckpt = json.loads(pretrain_json.read_text()).get("ckpt_path")
        if not ckpt or not Path(ckpt).exists():
            print(f"skip {d.name}: checkpoint {ckpt!r} missing")
            continue
        for hs in args.head_seeds:
            stem = probe_stem(args.dataset, hs)
            summary = d / f"{stem}.json"
            if summary.exists():
                print(f"skip {d.name} {args.dataset} head seed {hs}: {summary.name} exists")
                continue
            cmd = (
                [python, "finetune.py"]
                + to_cli({"dataset": args.dataset, "max_epochs": args.max_epochs})
                + ["--model_source", "checkpoint", "--model", str(ckpt), "--seed", str(hs)]
                + ["--experiment", EXPERIMENT, "--summary_json", str(summary)]
                # the encoder is frozen, so every head seed reuses one set of embeddings per run
                + ["--embed_cache", str(embed_cache(d, args.dataset))]
                + (["--random_init"] if random_init else [])
                + passthrough
            )
            if not args.dry_run:
                d.mkdir(parents=True, exist_ok=True)
            condor.append((d / f"{stem}.mlflow.db", cmd))
            jobs.append(
                lambda cmd=cmd, d=d, stem=stem: run_cmd(cmd, d / f"{stem}.log", args.dry_run, out / "mlflow.db")
            )
    if not jobs:
        print("probe: nothing to do")
        return
    if args.condor:
        # Head seeds of one run share --embed_cache and data/<dataset>; finetune.py file-locks both, so they can all
        # be queued at once (the first embeds, the rest wait and load)
        write_condor(args.condor, condor)
        return
    # The first probe processes data/<dataset> for this graph setup; run it alone so parallel jobs don't race on it
    rcs = [jobs[0]()] + run_all(jobs[1:], args.jobs)
    print(f"probe: {len(rcs)} launched, {sum(rc != 0 for rc in rcs)} failed")


def cmd_table(args, _passthrough):
    """Collect all jsons into results.csv (one row per pretraining run) and results.md (mean ± std over seeds).

    Every dataset in PROBE_METRICS with probe jsons in a run dir contributes ``<dataset>/<metric>`` columns (head seeds
    averaged within the run) and ``<dataset>/n_head_seeds``.
    """
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
        for ds, metrics in PROBE_METRICS.items():
            probes = [json.loads(f.read_text()) for f in sorted(d.glob(f"{probe_stem(ds, '*')}.json"))]
            row[f"{ds}/n_head_seeds"] = len(probes)
            for m in metrics:
                vals = [pr[m] for pr in probes if pr.get(m) is not None]
                row[f"{ds}/{m}"] = float(np.mean(vals)) if vals else None  # head seeds averaged per pretraining seed
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
    ]
    # one column per probe metric that any run has a value for
    cols += [
        (f"{ds}/{m}", label, 1, 3)
        for ds, metrics in PROBE_METRICS.items()
        for m, label in metrics.items()
        if pd.to_numeric(df[f"{ds}/{m}"], errors="coerce").notna().any()
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
        "\n".join(lines) + "\n\nmean ± std over pretraining seeds; probe metrics are first averaged over head seeds.\n"
    )
    (out / "results.md").write_text(md)
    print(md)
    print(f"wrote {out / 'results.csv'} and {out / 'results.md'}")


def _kept(d: Path) -> list:
    """(step, checkpoint path, sidecar record) for every checkpoint kept in run dir ``d`` (complete ones only).

    train.py writes a kept checkpoint's sidecar json after the checkpoint itself, so a sidecar marks it complete.
    The checkpoint is located next to its sidecar rather than by the recorded absolute path, so moved run dirs work.
    """
    out = []
    for sidecar in sorted((d / "checkpoints").glob("step_*.json")):
        record = json.loads(sidecar.read_text())
        ckpt = sidecar.with_suffix(".ckpt")
        if ckpt.exists():
            out.append((record["step"], ckpt, record))
    return out


def dynamics_runs(args) -> list:
    """Run dirs of the dynamics study for ``--config``, sorted by seed."""
    dyn = Path(args.out).resolve() / "dynamics"
    runs = [d for d in dyn.glob(f"{glob_escape(args.config)}_s*") if d.is_dir()]
    return sorted(runs, key=lambda d: int(d.name.rsplit("_s", 1)[1]))


def cmd_dynamics(args, passthrough):
    """Pretraining-dynamics study: keep checkpoints during pretraining, probe each, relate probes to pretraining loss.

    ``pretrain`` queues train.py with ``--keep_ckpt_steps``; ``probe`` queues finetune.py for every kept checkpoint
    found so far x dataset x head seed (re-run it as more checkpoints appear; finished probes are skipped);
    ``analyze`` writes dynamics.csv, dynamics_corr.csv, dynamics.md and plots into ``<out>/dynamics``.
    """
    {"pretrain": dynamics_pretrain, "probe": dynamics_probe, "analyze": dynamics_analyze}[args.stage](
        args, passthrough
    )


def dynamics_pretrain(args, passthrough):
    """One restart-safe train.py job per seed, keeping checkpoints at ``--keep_steps``."""
    cfg = select_configs([args.config])[0]
    dyn = Path(args.out).resolve() / "dynamics"
    python = "python" if args.condor else sys.executable
    jobs, condor = [], []
    for seed in args.seeds:
        d = run_dir(dyn, cfg["name"], seed)
        summary = d / "pretrain.json"
        if summary.exists():
            print(f"skip {d.name}: {summary.name} exists")
            continue
        cmd = (
            [python, "train.py"]
            + to_cli(PRESETS[args.preset])
            + to_cli(cfg)
            + ["--seed", str(seed), "--experiment", DYNAMICS["experiment"], "--summary_json", str(summary)]
            + ["--ckpt_dir", str(d / "checkpoints"), "--resume", "auto"]
            + ["--ckpt_every_n_steps", str(DYNAMICS["ckpt_every_n_steps"])]
            + ["--keep_ckpt_steps", *args.keep_steps]
            + passthrough
        )
        if not args.dry_run:
            d.mkdir(parents=True, exist_ok=True)
            config = {"preset": args.preset, **cfg, "seed": seed, "keep_steps": args.keep_steps}
            (d / "config.json").write_text(json.dumps(config, indent=2))
        condor.append((d / "pretrain.mlflow.db", cmd))
        jobs.append(lambda cmd=cmd, d=d: run_cmd(cmd, d / "pretrain.log", args.dry_run, dyn / "mlflow.db"))
    if args.condor:
        write_condor(args.condor, condor)
        return
    rcs = run_all(jobs, args.jobs)
    print(f"dynamics pretrain: {len(rcs)} launched, {sum(rc != 0 for rc in rcs)} failed")


def dynamics_probe(args, passthrough):
    """finetune.py for every kept checkpoint x dataset x head seed that has no summary json yet.

    Probes of checkpoint ``step_<N>`` live in ``<run>/probes/step_<N>/`` with one embedding cache per dataset there.
    """
    dyn = Path(args.out).resolve() / "dynamics"
    python = "python" if args.condor else sys.executable
    jobs, condor = [], []
    for d in dynamics_runs(args):
        for step, ckpt, _ in _kept(d):
            if args.steps and step not in args.steps:
                continue
            pd = d / "probes" / ckpt.stem
            for ds, hs in itertools.product(args.datasets, args.head_seeds):
                stem = probe_stem(ds, hs)
                summary = pd / f"{stem}.json"
                if summary.exists():
                    continue
                cmd = (
                    [python, "finetune.py"]
                    + to_cli({"dataset": ds, "max_epochs": args.max_epochs})
                    + ["--model_source", "checkpoint", "--model", str(ckpt), "--seed", str(hs)]
                    + ["--experiment", DYNAMICS["experiment"], "--summary_json", str(summary)]
                    + ["--embed_cache", str(embed_cache(pd, ds))]
                    + passthrough
                )
                if not args.dry_run:
                    pd.mkdir(parents=True, exist_ok=True)
                condor.append((pd / f"{stem}.mlflow.db", cmd))
                jobs.append(
                    lambda cmd=cmd, pd=pd, stem=stem: run_cmd(cmd, pd / f"{stem}.log", args.dry_run, dyn / "mlflow.db")
                )
    if not jobs:
        print("dynamics probe: nothing to do (no kept checkpoints yet, or every probe has its json)")
        return
    if args.condor:
        write_condor(args.condor, condor)
        return
    # The first probe of each dataset processes data/<dataset>; run one alone before the parallel rest
    rcs = [jobs[0]()] + run_all(jobs[1:], args.jobs)
    print(f"dynamics probe: {len(rcs)} launched, {sum(rc != 0 for rc in rcs)} failed")


def md_table(df, digits: int) -> str:
    """Markdown table of a DataFrame; floats to ``digits`` decimals, missing values as ``–``."""

    def cell(v):
        if v is None or (isinstance(v, float) and v != v):
            return "–"
        return f"{v:.{digits}f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|---" * len(df.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def dynamics_table(args):
    """One row per (run, kept step): sidecar pretraining metrics and probe metrics (mean/std over head seeds)."""
    import numpy as np
    import pandas as pd

    rows = []
    for d in dynamics_runs(args):
        name, seed = d.name.rsplit("_s", 1)
        for step, ckpt, record in _kept(d):
            row = {"config": name, "seed": int(seed), "step": step, "epoch": record.get("epoch")}
            row["final"] = step == record.get("planned_steps")
            row.update({k: record.get(k) for k in ["lr", *DYNAMICS_PRETRAIN]})
            pdir = d / "probes" / ckpt.stem
            for ds, metrics in PROBE_METRICS.items():
                probes = [json.loads(f.read_text()) for f in sorted(pdir.glob(f"{probe_stem(ds, '*')}.json"))]
                row[f"{ds}/n_head_seeds"] = len(probes)
                for m in metrics:
                    vals = [p[m] for p in probes if p.get(m) is not None]
                    row[f"{ds}/{m}"] = float(np.mean(vals)) if vals else None
                    row[f"{ds}/{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            rows.append(row)
    return pd.DataFrame(rows)


def dynamics_correlations(df):
    """Pearson and Spearman correlation of every downstream metric with every pretraining quantity and log10(step).

    Points are (run, kept step) pairs pooled over pretraining seeds; a pair needs both values. n < 3 gives NaN.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    df = df.assign(log10_step=np.log10(df["step"].where(df["step"] > 0)))
    rows = []
    for ds, metrics in PROBE_METRICS.items():
        for m, label in metrics.items():
            col = f"{ds}/{m}"
            if col not in df or df[col].isna().all():
                continue
            for x in [*DYNAMICS_PRETRAIN, "log10_step"]:
                pair = df[[x, col]].apply(pd.to_numeric, errors="coerce").dropna()
                r = {"dataset": ds, "metric": m, "label": label, "vs": x, "n": len(pair)}
                if len(pair) >= 3 and pair[x].nunique() > 1 and pair[col].nunique() > 1:
                    r["pearson_r"], r["pearson_p"] = stats.pearsonr(pair[x], pair[col])
                    r["spearman_rho"], r["spearman_p"] = stats.spearmanr(pair[x], pair[col])
                rows.append(r)
    return pd.DataFrame(rows)


def dynamics_plots(df, out: Path) -> list:
    """dynamics_vs_step.png (every metric against step, log x) and dynamics_vs_loss.png (probes against val losses)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    probe_cols = [
        (f"{ds}/{m}", label)
        for ds, metrics in PROBE_METRICS.items()
        for m, label in metrics.items()
        if f"{ds}/{m}" in df and df[f"{ds}/{m}"].notna().any()
    ]
    pre_cols = [(c, c) for c in ["val/noise_loss", "val/pred_loss", "val/pred_acc"] if df[c].notna().any()]
    paths = []

    panels = pre_cols + probe_cols
    ncol = min(3, len(panels))
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, (col, label) in zip(axes.flat, panels, strict=False):
        for seed, g in df.sort_values("step").groupby("seed"):
            std = g.get(f"{col}_std")
            ax.errorbar(g["step"], g[col], yerr=std, marker="o", ms=3, capsize=2, label=f"seed {seed}")
        positive = df["step"][df["step"] > 0]
        if len(positive):
            # symlog keeps a step-0 checkpoint on the axis; linear below the first positive step
            ax.set_xscale("symlog", linthresh=positive.min()) if (df["step"] == 0).any() else ax.set_xscale("log")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("pretraining step")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    paths.append(out / "dynamics_vs_step.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    xs = [c for c in ["val/noise_loss", "val/pred_loss", "val/loss"] if df[c].notna().any()]
    if probe_cols and xs:
        fig, axes = plt.subplots(
            len(probe_cols), len(xs), figsize=(3.8 * len(xs), 3.0 * len(probe_cols)), squeeze=False
        )
        color = np.log10(df["step"].clip(lower=1))
        for i, (col, label) in enumerate(probe_cols):
            for j, x in enumerate(xs):
                ax = axes[i, j]
                sc = ax.scatter(df[x], df[col], c=color, cmap="viridis", s=18)
                pair = df[[x, col]].dropna()
                if len(pair) >= 3 and pair[x].nunique() > 1 and pair[col].nunique() > 1:
                    rho = stats.spearmanr(pair[x], pair[col])[0]
                    ax.set_title(f"Spearman ρ = {rho:.2f} (n={len(pair)})", fontsize=9)
                ax.set_xlabel(x)
                ax.set_ylabel(label if j == 0 else "")
                ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=axes, label="log10 step", shrink=0.6)
        paths.append(out / "dynamics_vs_loss.png")
        fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
        plt.close(fig)
    return paths


def dynamics_analyze(args, _passthrough):
    """Write dynamics.csv, dynamics_corr.csv, dynamics.md and the plots for ``--config``'s runs."""
    import pandas as pd

    out = Path(args.out).resolve() / "dynamics"
    df = dynamics_table(args)
    if df.empty:
        sys.exit(f"no kept checkpoints under {out}/{args.config}_s*/checkpoints")
    df = df.sort_values(["seed", "step"])
    df.to_csv(out / "dynamics.csv", index=False)
    corr = dynamics_correlations(df)
    corr.to_csv(out / "dynamics_corr.csv", index=False)
    plots = dynamics_plots(df, out)

    probe_cols = [c for c in df.columns if "/" in c and c.split("/")[0] in PROBE_METRICS and "n_head" not in c]
    show = ["seed", "step", "val/noise_loss", "val/pred_loss", "val/pred_acc"]
    show += [c for c in probe_cols if not c.endswith("_std") and df[c].notna().any()]
    md = ["## Kept checkpoints", "", md_table(df[show], 3), ""]
    if not corr.empty and "spearman_rho" in corr:
        wide = corr.pivot_table(index="label", columns="vs", values="spearman_rho", sort=False)
        n = corr.groupby("label", sort=False)["n"].max()
        md += [
            "## Spearman ρ: downstream metric vs pretraining quantity",
            "",
            "Points are kept checkpoints pooled over pretraining seeds (n per row). A negative ρ against a loss means "
            "the downstream metric improves as the loss falls. Pearson r and p-values are in dynamics_corr.csv.",
            "",
            md_table(pd.concat([wide, n.rename("n")], axis=1).reset_index(names="metric"), 2),
            "",
        ]
    (out / "dynamics.md").write_text("\n".join(md))
    print("\n".join(md))
    print(
        f"wrote {out / 'dynamics.csv'}, {out / 'dynamics_corr.csv'}, {out / 'dynamics.md'}, "
        + ", ".join(map(str, plots))
    )


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

    for name, helptext in [("pretrain", "Run CONFIGS x seeds via train.py"), ("probe", "Downstream probe per run")]:
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
    sub.choices["probe"].add_argument(
        "--dataset", choices=list(PROBE_METRICS), default=PROBE["dataset"], help="finetune.py downstream dataset"
    )

    sub.add_parser("table", help="Aggregate jsons into results.csv / results.md")

    p = sub.add_parser(
        "dynamics", help="Downstream probes of checkpoints kept during pretraining", description=cmd_dynamics.__doc__
    )
    p.add_argument("stage", choices=["pretrain", "probe", "analyze"])
    p.add_argument("--config", default=DYNAMICS["config"], help="Config name from CONFIGS")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="Pretraining seeds (pretrain)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="scale", help="pretrain")
    p.add_argument(
        "--keep_steps", nargs="+", default=DYNAMICS["keep_steps"], help="train.py --keep_ckpt_steps (pretrain)"
    )
    p.add_argument("--datasets", nargs="+", choices=list(PROBE_METRICS), default=list(PROBE_METRICS), help="probe")
    p.add_argument("--head_seeds", type=int, nargs="+", default=[0, 1, 2], help="probe")
    p.add_argument("--steps", type=int, nargs="*", default=[], help="Probe only these kept steps (default: all)")
    p.add_argument("--max_epochs", type=int, default=PROBE["max_epochs"], help="probe")
    p.add_argument("--jobs", type=int, default=1, help="Concurrent subprocesses")
    p.add_argument("--dry_run", action="store_true", help="Print commands without running them")
    p.add_argument("--condor", default=None, help="Write an HTCondor queue file for hpc/gpu.sub instead of running")

    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if passthrough and args.command not in ("pretrain", "probe", "dynamics"):
        parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
    commands = {
        "loader": cmd_loader,
        "pretrain": cmd_pretrain,
        "probe": cmd_probe,
        "table": cmd_table,
        "dynamics": cmd_dynamics,
    }
    commands[args.command](args, passthrough)


if __name__ == "__main__":
    main()

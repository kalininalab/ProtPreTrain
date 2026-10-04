# Running STEP on the Saarland HPC (conduit, HTCondor)

The cluster only accepts containerised jobs (`universe = docker`). The image `ghcr.io/ilsenatorov/step` carries
**only the environment** (installed from `uv.lock`); the code is a git clone on the shared filesystem that every job
mounts. A code change needs a `git pull` on the cluster, not a rebuild. Rebuild the image (`hpc/build.sh`, on a
machine with docker) only when `pyproject.toml`/`uv.lock` change.

**The submit node has a 2 GB memory limit.** Run nothing heavier than `git`, `condor_*` and the stdlib-only queue
generator (`python3 scripts/benchmark.py ... --condor`) there. Everything else - data preparation, training,
aggregation, MLflow merging - is a job (`hpc/cpu.sub` or `hpc/gpu.sub`).

## Layout

```
/home/s8ilsena/step/          # `root` in hpc/common.sub (on /home: the group /scratch quota is ~full)
├── ProtPreTrain/             # git clone, branch hpc - submit from here
│   └── data -> ../data
├── data/                     # foldcomp databases (+ processed HDF5); downstream raw files in data/<task>/raw
├── runs/                     # your runs: checkpoints, per-job MLflow stores, summaries
├── bench/                    # scripts/benchmark.py --out
├── cache/                    # torch / HF / matplotlib caches
└── runlogs/                  # condor .out / .err / .log
```

## One-time setup

```bash
ssh conduit
git clone -b hpc git@github.com:kalininalab/ProtPreTrain.git /tmp/step-bootstrap
bash /tmp/step-bootstrap/hpc/setup_cluster.sh
```

Downstream datasets are not downloaded by the code. Copy their raw files to `data/<task>/raw/`, e.g. from a machine
logged in to W&B:

```bash
uvx wandb artifact get rindti/homology/homology_dataset:latest --root /tmp/homology
tar -xzf /tmp/homology/dataset.tar.gz -C /tmp/homology && rm /tmp/homology/dataset.tar.gz
rsync -a /tmp/homology/ conduit:/home/s8ilsena/step/data/homology/raw/
```

## Jobs

Two submit files, one queue-file format. Each line of a queue file is one job:

```
<mlflow store | ->, <command>
/home/s8ilsena/step/runs/demo/mlflow.db, python train.py --dataset m_jannaschii ... --ckpt_dir ... --resume auto
```

The first column is that job's own MLflow SQLite store (`-` for jobs that log nothing). One store per job because
SQLite locking is not safe across nodes on NFS. Commands must not contain commas: condor splits columns on them.

```bash
cd /home/s8ilsena/step/ProtPreTrain
condor_submit -a 'runfile=hpc/runs/smoke.txt' hpc/gpu.sub                 # GPU, image, mounts, network
condor_submit -a 'runfile=hpc/runs/prepare_afdb_rep_v4.txt' hpc/cpu.sub   # download + process before pretraining
condor_submit -a 'runfile=hpc/runs/pretrain_demo.txt' hpc/gpu.sub
condor_submit -a 'runfile=hpc/runs/finetune_demo.txt' hpc/gpu.sub         # after pretrain_demo finished
```

`condor_submit -a` inserts its argument just before the `queue` line, so it overrides any setting, e.g. a 4-GPU DDP
pretraining run: `-a 'request_GPUs=4' -a 'request_CPUs=32' -a 'request_memory=128G'`. Defaults are 1 GPU / 8 CPUs /
32 GB: modest asks match far more slots on this pool. Pass `--num_workers 7` (CPUs - 1) to train.py/finetune.py.

| Queue file | Submit with | What |
|---|---|---|
| `smoke.txt` | gpu.sub | `hpc/smoke.py`: versions, GPU, kernels, foldcomp server reachable |
| `prepare_<db>.txt` | cpu.sub | `scripts/prepare_data.py`: download + process a foldcomp DB, cache lengths |
| `pretrain_demo.txt` | gpu.sub | small restart-safe pretraining run on m_jannaschii |
| `finetune_demo.txt` | gpu.sub | fluorescence probe of the demo checkpoint, 2 head seeds |
| `aggregate.txt` | cpu.sub | `scripts/aggregate_results.py` over every store -> `results.csv` |
| `bench_table.txt` | cpu.sub | `scripts/benchmark.py table` |
| `merge_mlflow.txt` | cpu.sub | merge every store into `merged.db` for `mlflow ui` |

### The benchmark matrix

`scripts/benchmark.py` writes queue files instead of running when given `--condor` (stdlib only - fine on the
submit node). The probe file needs the finished pretraining runs' `pretrain.json`, so generate it afterwards:

```bash
python3 scripts/benchmark.py --out /home/s8ilsena/step/bench pretrain --preset scale --seeds 0 1 2 \
    --condor hpc/runs/bench_pretrain.txt -- --num_workers 7
condor_submit -a 'runfile=hpc/runs/bench_pretrain.txt' hpc/gpu.sub
# ...once they have finished:
python3 scripts/benchmark.py --out /home/s8ilsena/step/bench probe --condor hpc/runs/bench_probe.txt -- --num_workers 7
condor_submit -a 'runfile=hpc/runs/bench_probe.txt' hpc/gpu.sub
# other downstream datasets (homology is the default): fluorescence, stability, deeploc
python3 scripts/benchmark.py --out /home/s8ilsena/step/bench probe --dataset stability \
    --condor hpc/runs/bench_probe_stability.txt -- --num_workers 7
condor_submit -a 'runfile=hpc/runs/bench_probe_stability.txt' hpc/gpu.sub
condor_submit -a 'runfile=hpc/runs/bench_table.txt' hpc/cpu.sub   # -> bench/results.md
```

Homology probes write `probe_h<seed>.{json,log,mlflow.db}` and cache embeddings in `<run>/embeddings/`; any other
dataset writes `probe_<dataset>_h<seed>.*` and `<run>/embeddings_<dataset>/`. `table` reports every dataset that has
results (metrics per dataset: `PROBE_METRICS` in the script). A run whose summary json already exists is skipped, so
regenerating a queue file after failures only resubmits what is missing. `--preset scale` needs
`prepare_afdb_rep_v4.txt` to have finished first.

## Restarts

Pretraining queued through these files uses `--ckpt_dir <run>/checkpoints --resume auto`: `last.ckpt` is refreshed at
every epoch end (and every `--ckpt_every_n_steps` steps if set), and a rerun of the same command continues from it in
the same MLflow run. `gpu.sub` sets `max_retries = 2`, so a job that crashes is restarted automatically; one that is
removed can simply be resubmitted.

## Results and MLflow

Each job writes its own store. To read them:

- `aggregate.txt` (cpu job) -> `/home/s8ilsena/step/results.csv`, mean ± 95% CI over seeds.
- Browse in the MLflow UI on your own machine:

  ```bash
  rsync -a --include='*/' --include='*.db' --exclude='*' conduit:/home/s8ilsena/step/ step_stores/
  python scripts/merge_mlflow.py --src 'step_stores/*/**/*.db' --dest sqlite:///merged.db
  mlflow ui --backend-store-uri sqlite:///merged.db
  ```

## Monitoring

```bash
condor_q -nobatch                 # your jobs
condor_q -better-analyze <id>     # why is it idle?
condor_tail -f <id>               # follow stdout
condor_ssh_to_job <id>            # shell inside the running container
condor_rm <id>
tail -f ../runlogs/gpu.<cluster>.<proc>.out
```

## Gotchas

- **Never add `environment = ...` to a submit file.** The cluster injects `HOME` only when it is unset, and the home
  mount depends on it. Put variables in `hpc/job.sh`.
- **`GPUs_Capability >= 7.5` is load-bearing.** The cu130 torch wheels carry sm_75..sm_120 (A100, L40S, H200,
  Blackwell); the P100/V100 nodes would fail at the first kernel launch. They also need a CUDA 13 driver, which every
  node has (2026-10).
- **At most 150 jobs per `condor_submit`** (`SUBMIT_REQUIREMENT_MaxMaterializations`). Split bigger queue files.
- **HTTPS to github.com fails from the submit nodes**; clone and pull over SSH.
- **Shared data is prepared once.** Concurrent jobs that need the same foldcomp database, downstream dataset or
  `--embed_cache` file-lock it: one prepares, the rest wait. Still, run `prepare_<db>.txt` before a big pretraining
  batch so GPU jobs don't sit idle waiting for a download.
- **Images are pinned by content tag** (`env-<hash of uv.lock + hpc/Dockerfile>`) in `hpc/common.sub`;
  `hpc/build.sh` updates that line after pushing - commit it. Never point jobs at `:latest`: execute nodes cache
  images and do not re-pull a tag they already have, so a rebuilt `:latest` keeps running the old image there.
  The first pull of a new image on a node takes a few minutes (several GB).
- **Triton needs a C compiler at runtime** (torch 2.14 JIT-compiles some eager-mode kernels), so the image carries
  `gcc`; without it training dies at the first backward pass.
- **The group `/scratch` quota (10 TiB, shared) was at 9.84 TiB on 2026-10-04.** A full quota refuses every write,
  even a symlink. Check `/scratch/chair_kalinina/.quota_info` before moving `root` there.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## What this is

STEP: self-supervised pretraining of a graph transformer on protein structures (AlphaFold DB via foldcomp),
then frozen-embedding evaluation on downstream tasks. Everything is PyTorch Lightning + PyTorch Geometric.
MLflow tracks experiments (params + metrics only); checkpoints and datasets are plain files on local disk.
The tracking store is `$MLFLOW_TRACKING_URI`, defaulting to `sqlite:///mlflow.db` in the working directory
(`step/utils/tracking.py`); browse it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Commands

```bash
# Environment: uv-managed (Python 3.11, torch 2.14+cu130 - needs a CUDA 13 driver, pinned in uv.lock); creates .venv/
uv sync

# Pretraining (uses every GPU visible to the process)
python train.py --dataset afdb_rep_v4 --hidden_dim 512 --num_layers 12 --masktype normal --maskfrac 0.15
# Small end-to-end smoke run (~30 s on one GPU): m_jannaschii is the smallest foldcomp DB (8 MB, 1773 structures)
python train.py --dataset m_jannaschii --subset 256 --val_size 32 --hidden_dim 192 --num_layers 2 --max_epochs 1

# Downstream eval: freeze an encoder, embed the dataset, train an MLP head (raw data must be in data/<task>/raw)
python finetune.py --dataset fluorescence --model_source checkpoint --model checkpoints/<run_id>/<file>.ckpt
python finetune.py --dataset homology --model_source ankh --model ankh-base
python finetune.py --dataset stability --model_source prostt5 --model Rostlab/ProstT5 --ablation sequence

# Aggregate downstream test metrics over seeds (mean ± 95% CI) and pretraining GPU-hours from MLflow
python scripts/aggregate_results.py --out results.csv

# Smoke tests (CPU, ~10 s): imports, forward/predict/training steps for the rw / seq / invariant configs,
# rotation invariance, masking transforms, MLflow logging, and a seeded loss snapshot
python -m pytest

# Lint / format (pre-commit: ruff + ruff-format at line-length 119, plus interrogate)
pre-commit install
pre-commit run --all-files
```

Tests live in `tests/test_smoke.py`; `test.py` and `test*.ipynb` are scratch/benchmark files, not tests.
`tests/snapshots/baseline.pt` pins seeded training losses/outputs for each model config — a deliberate change to
the model or loss means deleting it and rerunning `pytest` to regenerate. `interrogate` in pre-commit rejects
commits below 80% docstring coverage, so new public functions/classes need a docstring.

## Architecture

**Pretraining path** — `train.py` → `FoldCompDataModule` → `FoldCompDataset` → `DenoiseModel`.

- `step/data/datasets.py:FoldCompDataset` wraps a foldcomp DB (downloaded on first use into `data/<db_name>/raw`).
  Processing is chunked: `chunk_size` (4096) structures per HDF5 file, written in parallel with joblib, and
  `get(idx)` reads back from `chunk_{idx // chunk_size * chunk_size}.h5`. This is an on-disk `Dataset`, not
  `InMemoryDataset`; deleting `data/<db>/processed` forces reprocessing (expensive).
- Structures become graphs in `step/data/parsers.py:ProtStructure.get_graph()` (residue nodes, `x` = AA type,
  `pos` = coordinates).
- Two transform tiers, and which tier a transform belongs to matters:
  - `pre_transforms` run once at process time and are baked into the HDF5 chunks — `Center`, `NormalizeRotation`.
  - `transforms` run per-sample at load time — `PosNoise` (adds Gaussian coordinate noise, stored as `noise`), one of
    `MaskType` / `MaskTypeAnkh` / `MaskTypeBERT` (corrupts `x`, keeps ground truth in `orig_x`/`mask`), then
    `graph_transforms(radius, pe, walk_length)` (`RadiusGraph`, `ToUndirected`, plus `RandomWalkPE` only for `--pe rw`)
    so the graph comes from the noisy coordinates.
  - Downstream datasets bake only `Center`/`NormalizeRotation` into `pre_transform`; the graph is built by
    `graph_transforms` at embedding time from the *checkpoint's* hparams (radius, pe), so processed files never go stale.
- `--pe`: `seq` (default; sinusoidal residue index, computed in-model), `none`, or `rw` — `RandomWalkPE`, a dense
  O(N³·walk_length) CPU transform in the dataloader workers, ~13× slower loading (~65× near 1000 residues).
- `step/models/denoise.py:DenoiseModel` — embeds AA type + raw coordinates (`pos_dim`) + PE into `hidden_dim`
  (`hidden_dim > pe_dim + pos_dim` is asserted), then `num_layers` GPSConv blocks (GINConv local, or GINEConv with
  `--edge_dim` RBF edge-length features, + performer/multihead global attention), with two heads: noise regression
  (MSE) and residue-type classification (CE). `--invariant true` drops the raw-coordinate input and uses
  `EquivariantNoiseHead` (noise = learned weighting of unit vectors to neighbours), making embeddings rotation-invariant
  and the noise prediction rotation-equivariant; it requires `edge_dim > 0`. Constructor defaults reproduce the
  original architecture so old checkpoints load; `train.py` holds the new defaults.
- `--scheduler cosine` (default) warms up for `--warmup_frac` of `trainer.estimated_stepping_batches`, then one cosine
  decay to the end; `legacy` is the old fixed 10k-step warmup / 100k-step cycle, which never leaves warmup in short
  runs and restarts every 200k steps in long ones. Loss = `alpha * noise_loss + (1 - alpha) * type_loss`; `predict_all` switches between
  predicting types for all nodes vs. only masked ones. `RedrawProjection` periodically redraws performer
  projection matrices — required for performer attention correctness.
- Checkpoints are written by a plain Lightning `ModelCheckpoint` to `checkpoints/<mlflow run_id>/`; `--summary_json`
  records the path, and `finetune.py --model_source checkpoint --model <path>` consumes it. Nothing is uploaded.
- Loggers come from `step/utils/tracking.py:mlflow_logger()`, which creates the store and experiment under a file lock:
  concurrent first runs on a fresh SQLite store otherwise race on MLflow's schema migration and all fail.
  `train.py` logs hyperparameters only through `model.hparams` (every CLI arg arrives via `**kwargs`); also logging
  `vars(args)` would fail, because MLflow rejects changing a param and the model rewrites some (e.g. `pe_dim=0`).

**Downstream path** — `finetune.py` → a `DownstreamDataModule` subclass → a head in `step/models/downstream.py`.

The key design point: the encoder is **never** fine-tuned. `DownstreamDataModule.setup()` builds the dataset,
then `embed_splits()` runs the frozen feature extractor over every split up front and replaces the split with a
plain list of `Data` objects whose `x` is the embedding. Only a small MLP head then trains.
`feature_extract_model_source` selects the extractor (local `checkpoint` = our `DenoiseModel`,
plus `huggingface`, `ankh`, `prostt5` baselines) and *also* controls whether graph transforms are applied at all — the
other sources are sequence models and get `transform = pre_transform = None`.

- Downstream datasets (`FluorescenceDataset`, `StabilityDataset`, `HomologyDataset`, `DTIDataset`) are
  `InMemoryDataset`s read from raw files placed by hand in `data/<task>/raw/` (nothing is downloaded; a missing
  file raises an error naming the files and the one-time fetch of the original W&B artifact
  `rindti/<task>/<task>_dataset`), one processed `.pt` per split. Fluorescence is special: it has a single GFP structure and derives per-mutant graphs via
  `compute_edits`/`apply_edits` in `step/data/utils.py`. Homology has three test splits
  (fold/superfamily/family), which is why `HomologyModel.test_step` dispatches on `dataloader_idx`.
- Heads use `LazySimpleMLP` (LazyLinear) so embedding dimension does not need to be known in advance.
- `--ablation sequence|structure` injects `SequenceOnly`/`StructureOnly` transforms to zero out one modality,
  and only applies to our `checkpoint` model.

## Gotchas

- `train.py` parses args *before* importing torch (deliberate — keeps `--help` fast); keep new imports below
  `args = parser.parse_args()`.
- `train.py`/`finetune.py` use `accelerator="auto"` and every visible GPU; the repo is cluster-agnostic, cluster launch
  tooling lives on a separate branch.
- The graph (radius edges + optional RandomWalkPE) is built at *load time after `PosNoise`*; building it in `pre_transforms` leaks
  the clean structure into the denoising target (kept only as the `--clean_graph` ablation).
- `--batch_sampling` needs `use_distributed_sampler=False` (set in `train.py`): `DynamicBatchSampler` shards across DDP
  ranks itself, from `FoldCompDataset.lengths()`.
- `data/` holds real processed datasets and is gitignored — do not delete it casually.
- Precision is `bf16-mixed` only when CUDA is available: CPUs without native bf16 run it ~8× slower than fp32.
- `scripts/benchmark.py` runs the pretraining ablation matrix (`CONFIGS`) and the frozen-embedding homology probe;
  `--preset local` uses `data/e_coli_bench` (symlinked E. coli proteome), `--preset scale` is the cluster setup.
  Bench runs log to their own store, `<out>/mlflow.db` (experiment `step-bench`).

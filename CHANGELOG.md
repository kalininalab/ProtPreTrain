# CHANGELOG

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [hpc branch] — 2026-10-04 — running on the conduit HTCondor cluster

### Added

- `hpc/`: environment-only docker image (`hpc/Dockerfile`, `hpc/build.sh` →
  `ghcr.io/ilsenatorov/step`), job wrapper, `gpu.sub`/`cpu.sub`, setup script, smoke
  test, example queue files and `hpc/README.md`.
- `train.py --ckpt_dir`, `--resume auto`, `--ckpt_every_n_steps`: restart-safe
  pretraining that continues the same MLflow run.
- `scripts/prepare_data.py`: download + process a foldcomp database as its own job.
- `scripts/merge_mlflow.py` and `aggregate_results.py --db_glob`: one MLflow store
  per job (SQLite locking is unsafe across NFS clients), read together.
- `scripts/benchmark.py pretrain|probe --condor FILE`: write HTCondor queue files.

### Changed

- torch 2.14 / pyg-lib now use the CUDA 13.0 build (`cu130`): the only torch 2.14
  wheels with Blackwell (sm_120) kernels. Needs a CUDA 13 driver (>= 580); sm_75+.
- Shared dataset processing, the lengths cache and `--embed_cache` are file-locked,
  so concurrent jobs prepare them once.
- `finetune.py` checkpoints go to `checkpoints/<run_id>` (concurrent jobs collided).
- Progress bar: Rich on a terminal, sparse tqdm in batch logs.
- The root `Dockerfile` moved to `hpc/Dockerfile` (environment-only).

### Fixed

- `FoldCompDataset` required `<db>.source`, which only `afdb_rep_v4` has, so every
  other database re-downloaded on each load.

## [Unreleased] — 2026-10-04 — Weights & Biases replaced by MLflow

MLflow now tracks experiments (params and metrics only). Checkpoints and datasets
are plain files on local disk; nothing is uploaded or downloaded. `wandb` is no
longer a dependency.

### ⚠️ Breaking / behaviour-changing

- **Tracking store:** `$MLFLOW_TRACKING_URI`, defaulting to `sqlite:///mlflow.db` in
  the working directory. View with `mlflow ui --backend-store-uri sqlite:///mlflow.db`;
  point at a shared server by setting the env var.
- **`--wandb_project` → `--experiment`** on `train.py` (default `step`) and
  `finetune.py` (default: the dataset name).
- **`finetune.py --model_source wandb` is gone.** Pass a local checkpoint:
  `--model_source checkpoint --model checkpoints/<run_id>/<file>.ckpt`. An old W&B
  checkpoint can be fetched once with
  `uvx wandb artifact get rindti/step/model-<runid>:latest --root <dir>`.
- **Downstream datasets are no longer downloaded.** Their raw files must be in
  `data/<task>/raw/`; if any are missing, the dataset raises a `FileNotFoundError`
  listing them and the one-time fetch command for the original W&B artifact.
- **Checkpoints are no longer uploaded**; a plain `ModelCheckpoint` writes them to
  `checkpoints/<mlflow run_id>/` (same layout as before).
- `scripts/aggregate_results.py` reads MLflow (`--tracking_uri`, `--experiments`)
  instead of the W&B API; GPU-hours come from the logged `train_time_s × world_size`.
  Results that exist only in W&B are not included.

### Added

- `step.utils.mlflow_logger()`: builds the `MLFlowLogger` after creating the store and
  experiment under a file lock (`<db>.lock`). Without it, concurrent first runs on a
  fresh SQLite store raced on MLflow's schema migration and all failed.
- `train.py` logs `num_params`, `world_size` and `train_time_s` as metrics.
- Tests: MLflow logging of a short fit; the missing-dataset error.

### Changed

- `scripts/benchmark.py` logs to `<out>/mlflow.db` (experiment `step-bench`).
- `mlflow` 3.16.1 and `filelock` added; `wandb` removed. mlflow's `databricks-sdk`
  dependency caps protobuf below 7 (7.36.1 → 6.33.6).

## [Unreleased] — 2026-10-03 — `dev` branch ported in

`dev` and this branch were independent rewrites of `69960b4`. The useful parts of
`dev` were ported one commit at a time (each commit names its `dev` source);
nothing here changes the training objective or invalidates checkpoints.

### ⚠️ Breaking / behaviour-changing

- **Environment is now uv-managed**: `pyproject.toml` + `uv.lock` (Python 3.11,
  torch 2.14+cu126, PyG 2.8, lightning 2.6.6, wandb 0.30, foldcomp 1.0). `setup.py`
  and `requirements.txt` are gone; `requirements-asrun.txt` stays as the record of
  the environment that produced the original results. Install with `uv sync`.
- **`lightning.pytorch` namespace** replaces `pytorch_lightning`. Checkpoints saved
  under `pytorch_lightning` load unchanged (verified).
- **`--masktype ankh` masking now uses the torch RNG** (deterministic under a seed)
  and always fills its quota. Previously, graphs with fewer distinct residue types
  than the quota (with quota < 20) were under-masked.
- `DenoiseModel` constructor checks raise `ValueError` instead of `AssertionError`.

### Added

- `tests/test_smoke.py` (`python -m pytest`, ~10 s on CPU): forward / predict /
  training steps for the rw, seq and invariant configs, rotation invariance of the
  invariant model, masking transforms (boolean masks, full-length `orig_x`, after
  batching), and a seeded snapshot (`tests/snapshots/baseline.pt`).
- uv-based `Dockerfile`.

### Fixed

- `train.py` crashed under wandb 0.30 (`wandb.Settings(start_method=...)` removed).
- `DenoiseModel.forward` no longer overwrites the input batch's `x` with embeddings.
- ProstT5 embedding only skips CUDA OOM (with a warning) instead of every `RuntimeError`.
- `RandomWalkPE` no longer allocates an N×N zero matrix.
- `FoldCompDataModule` no longer uses mutable list defaults.

### Changed

- ruff + ruff-format replace black, isort and flake8 (interrogate stays).
- `ankh`/`transformers` are imported only when those baselines are used.
- Removed dead code: `ToCuda`, `ToCpu`, `MaskTypeWeighted`, figure logging
  (`step/utils/vis.py`), `step/utils/math.py`, unused `cli.py` and data helpers.

### Not ported from `dev`

- PyG's `DynamicBatchSampler` (this branch's DDP-safe sampler supersedes it).
- `SequentialLR` scheduler (this branch's `--scheduler cosine|legacy` keeps old
  checkpoints loadable).
- `dev`'s `MaskTypeAnkh`/`MaskTypeBERT` index masks and subset `orig_x` (the batching
  bug fixed on this branch).

## [Unreleased] — 2026-09-29

Pre-publication pass on experimental validity and portability. **Every pretrained
checkpoint produced before this must be retrained** (items 1 and 3).

### ⚠️ Breaking / behaviour-changing

1. **Denoising no longer leaks the clean structure.** The radius graph and
   random-walk PE were baked into the processed chunks from clean coordinates while
   `PosNoise` ran at load time, so the model saw noise-free connectivity while
   predicting the noise. They are now built at load time after `PosNoise`.
   `--clean_graph true` restores the old ordering as an ablation. Existing processed
   chunks stay valid; PyG warns once that the pre-transform changed.
2. **The sequence-only ablation is now sequence-only.** `SequenceOnly` runs before the
   radius graph and PE, so they come from the straight-line positions rather than the
   true structure. Its processed files go to `processed_sequence/`.
3. **`predict_all` defaults to `False`** (`train.py` and `DenoiseModel`). With `True`
   the type loss was dominated by unmasked residues whose type is in the input.
4. **Pretraining drops structures longer than 1022 residues** (`--max_length`, as in
   ESM). Lengths are read from HDF5 shapes and cached in `processed/lengths.npy`.

### Added

- `finetune.py --random_init`: no-pretraining control, same architecture, fresh weights.
- `train.py --sequence_only`: compute-matched sequence-only pretraining control.
- `--seed` on `train.py` and `finetune.py` (finetune previously set none).
- `finetune.py --ablation_maskfrac`: masking rate for `--ablation structure`.
- `scripts/aggregate_results.py`: mean and 95% CI across seeds from wandb, plus
  pretraining GPU-hours (`train.py` now logs `num_params` and `world_size`).

### Fixed

- `DynamicBatchSampler` rewritten. It loaded (and transformed) every sample in the
  main process to size batches, repeated samples after skipping an oversized one, and
  under DDP each rank yielded empty batches after exhausting its shard. It now batches
  from cached lengths and shards batches evenly across ranks.
- `train.py` crashed at `trainer.fit`: `setup()` was called manually and again by
  Lightning, and `FoldCompDataset.__init__` called `torch.set_num_interop_threads`,
  which raises on a second call. Both removed; one-time processing moved to
  `prepare_data()` so DDP ranks don't race on first use.
- `num_workers=0` passed `n_jobs=0` to joblib in dataset processing.

### Changed

- Cluster-agnostic: `train.py` and `finetune.py` use `accelerator="auto"` and all
  visible GPUs instead of hardcoding 4.
- Removed the SLURM job scripts, `scripts/wandb_sync.sh`, the stale `Dockerfile` and
  the stale `config/*.yaml`.

## [2026-08-19] — maintenance pass

Maintenance and pre-publication pass: dependency audit/upgrade (Phase 1) and bug
audit (Phase 2). No model architecture or training-objective changes were made
beyond the bug fixes listed below.

### ⚠️ Breaking / behaviour-changing

- **Pretraining data is now shuffled.** `shuffle=True` on the datamodule was
  silently ignored (see Fixed #5), so every previous pretraining run consumed the
  dataset in fixed on-disk order. Runs will no longer reproduce prior results
  bit-for-bit, and random reads across HDF5 chunks are slower than the sequential
  reads this code was previously getting. Pass `shuffle=False` to
  `FoldCompDataModule` to restore the old behaviour exactly.
- **Frozen embeddings change.** `DenoiseModel.predict_step` previously skipped
  BatchNorm on the random-walk positional encoding (Fixed #1). Every downstream
  embedding produced before this fix was computed inconsistently with training.
  **Downstream results must be regenerated.**
- **`--masktype ankh` targets change.** Mask indices were not offset during graph
  batching (Fixed #2), so with `--predict_all False` the type-prediction loss was
  computed against wrong targets for every graph after the first in each batch.
  Any `--masktype ankh --predict_all False` run is invalid and must be rerun.
- **Custom transforms must now define `forward()`, not `__call__()`** — required by
  PyG 2.8, which made `BaseTransform` an ABC. Third-party transforms subclassing
  these will need the same change.
- **`batch.mask` is now always a boolean mask** (`[num_nodes]`), never an index
  tensor. `batch.orig_x` is now always full-length (`[num_nodes]`) for every
  masking transform. Downstream code indexing these must be updated.

### Added

- `requirements.txt`: exact pins for the full verified environment, plus the
  two-step install order the PyG binary wheels require.
- `requirements-asrun.txt`: snapshot of the original (pre-repair) environment,
  preserved as the record of what produced the existing results.
- `pyg-lib==0.7.0` — **newly required**: PyG 2.8 moved `radius_graph` (used by
  `T.RadiusGraph` in the pre-transform pipeline) off `torch-cluster` onto pyg-lib.
- `h5py`, `rdkit`, `joblib`, `tqdm`, `matplotlib`, `plotly`, `pyyaml`, `beartype`,
  `scipy`, `sentencepiece` to `requirements.txt` — all were imported by the code
  but undeclared. `h5py` (`step/data/datasets.py`) and `rdkit`
  (`step/data/utils.py`) were also *not installed*, so `import step` failed
  outright before this pass.

### Changed — dependencies

Upgraded, each verified with a full smoke test (data → transforms → batching →
forward → loss) before proceeding to the next:

| Package | From | To |
|---|---|---|
| torch | 2.2.2 (conda, cpu) | 2.9.0+cpu (pip) |
| torch_geometric | 2.5.2 | 2.8.0.post1 |
| pyg-lib | *(absent)* | 0.7.0+pt29cpu |
| torch-cluster / -scatter / -sparse | ABI-broken | 1.6.3 / 2.1.2 / 0.6.18 (+pt29cpu) |
| lightning / pytorch-lightning | 2.4.0 | 2.6.5 |
| torchmetrics | 1.4.1 | 1.9.0 |
| transformers | 4.44.2 | 4.57.6 |
| numpy | 1.26.4 | 2.4.6 |
| scipy | 1.13.1 | 1.17.1 |
| pandas | 2.2.2 | 3.0.5 |
| wandb | 0.18.0 | 0.28.2 |

**Root cause of the broken environment:** the env mixed a conda-channel CPU build
of `pytorch` (`pytorch-mutex cpu`) with pip-installed PyG extension wheels built
against the CUDA/PyPI torch. The resulting C++ ABI mismatch
(`undefined symbol: _ZN3c1017RegisterOperatorsD1Ev`) made `import torch_geometric`
fail outright. Fixed by sourcing torch and all PyG extensions from pip with
matching build tags.

**Removed from `requirements.txt`** (declared but never imported): `yacs`, `ogb`,
`tensorboardX`, `performer-pytorch`. Performer attention comes from PyG's built-in
`attn_type="performer"`, not the standalone package.

**Not upgraded, deliberately:**

- `transformers` capped `<5`. transformers 5 removes the TensorFlow classes, and
  `ankh` 1.10.0 (latest release; no newer version exists) imports
  `TFT5EncoderModel` at module load. Verified: transformers 5.15.1 breaks
  `import step.data.datamodules`, disabling all of `finetune.py`. Do not bump
  without replacing or vendoring `ankh`.
- `foldcomp` held at 0.0.7 (1.0.0 is a major bump; the data path it governs
  cannot be exercised without a downloaded AFDB database).

### Fixed

1. **`step/models/denoise.py:161` — `predict_step` discarded PE normalization.**
   Computed `pe = self.pe_norm(batch.pe)` and then immediately overwrote it with
   `self.pe_encode(batch.pe)`, feeding the *raw* positional encoding to the
   encoder while `forward()` (L93–94) correctly feeds the normalized one.
   Changed to `pe = self.pe_encode(pe)`.
   *Why it matters:* `predict_step` produces every frozen embedding consumed by
   downstream evaluation, so all downstream numbers came from a model evaluated
   inconsistently with how it was trained. **Highest-impact fix in this pass.**

2. **`step/data/transforms.py:90–112` — `MaskTypeAnkh` mask broke under batching.**
   Stored `batch.mask` as a tensor of node *indices*. PyG concatenates unknown
   attributes without adding per-graph node offsets, so after collation the
   indices addressed the wrong graphs. Measured on a 3-graph batch: only **9 of 27**
   indices pointed at genuinely masked nodes. Silent — no exception raised.
   Now emits a boolean mask, which collates correctly.

3. **`step/data/transforms.py:123–135` — `MaskTypeBERT` stored a subset `orig_x`.**
   Set `batch.orig_x = batch.x[indices].clone()` while `training_step` treats
   `orig_x` as full-length, giving `ValueError: Expected input batch_size (128) to
   match target batch_size (18)` with `--predict_all True` and `IndexError` with
   `False`. `--masktype bert` could never run in either mode. Now stores a
   full-length `orig_x` plus a boolean mask.

4. **`step/data/transforms.py:141–153` — `MaskTypeWeighted`** had the identical
   subset-`orig_x` defect. Fixed the same way. (Latent: not reachable from the
   `train.py` CLI, which offers only `normal|ankh|bert`.)

5. **`step/data/datamodules.py:93` — `shuffle` never reached the DataLoader.**
   `_dl_kwargs(shuffle)` returned `self.shuffle if shuffle else False`, and every
   call site passed `False`, so `shuffle=True` was unconditionally discarded and
   training data was never shuffled. `_get_dataloader` now takes an explicit
   `shuffle` argument and `train_dataloader` passes `True`; validation and test
   remain unshuffled. See the behaviour-changing note above.

6. **`step/data/transforms.py:19` and `step/data/datamodules.py:135` — device
   placement.** `RandomWalkPE` defaults to `cuda=True`, and the downstream
   pre-transform pipeline never overrode it, so `finetune.py --model_source wandb`
   died with `AssertionError: Torch not compiled with CUDA enabled` on any
   CPU-only host. `RandomWalkPE` now falls back to CPU when CUDA is unavailable,
   and the datamodule passes `cuda=torch.cuda.is_available()` explicitly. GPU
   behaviour is unchanged.

7. **`step/models/downstream.py:82` — removed `verbose=True` from
   `ReduceLROnPlateau`.** torch 2.9 removed the argument; `configure_optimizers`
   raised `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword
   argument 'verbose'`, breaking every downstream model at trainer setup.

8. **`step/data/datasets.py:135` — `torch.load` under torch ≥ 2.6.** torch 2.6
   flipped the `weights_only` default to `True`, which refuses to unpickle the
   PyG `Data`/`slices` objects these archives hold. Now passes
   `weights_only=False` explicitly. Without it, every downstream dataset fails to
   load with `UnpicklingError`.

9. **`step/data/transforms.py` — `__call__` → `forward` for PyG 2.8.**
   `PosNoise`, `MaskType`, `MaskTypeAnkh`, `MaskTypeBERT` and `MaskTypeWeighted`
   overrode `__call__`; PyG 2.8 made `BaseTransform` an ABC with an abstract
   `forward()`, so these became uninstantiable
   (`TypeError: Can't instantiate abstract class MaskType`). Side benefit: PyG's
   `__call__` shallow-copies before dispatching, so these transforms no longer
   mutate the caller's `Data` in place.

### Investigated, not changed

Confirmed correct — recorded so they are not "fixed" into bugs later:

- `len(Batch) == num_graphs` in PyG 2.5 and 2.8, so the iteration in
  `datamodules._embed_with_wandb` and `DTIModel.forward` is correct.
- `RandomWalkPE`'s zero-filled first PE channel correctly equals `diag(A¹) = 0`
  for loop-free graphs, matching PyG's reference implementation.
- Per-graph embeddings stored as 1-D `x` of shape `[hidden]` collate to
  `[B*hidden]`, which `to_dense_batch` correctly recovers as `[B, hidden]`.
  Fragile but correct.
- `HomologyModel.test_step`'s `["fold", "superfamily", "family"]` ordering matches
  the dataloader order returned by `HomologyDataModule.test_dataloader`.

### Verification

- Smoke test: graph → `Center`/`NormalizeRotation`/`RadiusGraph`/`ToUndirected`/
  `RandomWalkPE` → `PosNoise` + mask → batch → `DenoiseModel.forward` → loss,
  across all 3 mask types × `predict_all` ∈ {True, False}.
  **Before: 4 passed, 2 failed. After: 6/6 pass.**
- Downstream heads (`RegressionModel`, `ClassificationModel`, `HomologyModel`,
  `DTIModel`): 4/4 pass. `configure_optimizers`: 5/5 pass.
- Mask-batching correctness assertion: `MaskTypeAnkh` 9/27 → 27/27 correct.
- All 17 project modules import; `pip check` reports no broken requirements.
- `interrogate` docstring coverage 91% (gate: 80%). `black --check` clean on all
  modified files.

### Known issues (not addressed in this pass)

- No LICENSE file — blocks publication. Awaiting a decision on which license.
- `train.py:51` passes `wandb.Settings(start_method="fork")`, which wandb 0.28
  reports as deprecated and non-functional.
- `step/models/downstream.py:133` uses `torch.tensor(batch.y)` on an existing
  tensor, which warns; `.detach().clone()` is the recommended form.
- No test suite. `test.py`, `test.ipynb`, `test2.ipynb` are scratch/benchmark
  files despite the pytest config in `pyproject.toml`.

## [v1.0.0]

- Repo setup

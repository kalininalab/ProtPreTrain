# CHANGELOG

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — 2026-08-19

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
- `config/*.yaml` reference `FoldSeekDataModule`, a class that no longer exists;
  nothing loads them.
- `train.py` hardcodes `devices=4, accelerator="gpu"`; `finetune.py` hardcodes
  `devices=-1`. Neither runs on a different machine without editing.
- `finetune.py` sets no seed (`train.py` does: `pl.seed_everything(42)`).
- `train.py:51` passes `wandb.Settings(start_method="fork")`, which wandb 0.28
  reports as deprecated and non-functional.
- `step/models/downstream.py:133` uses `torch.tensor(batch.y)` on an existing
  tensor, which warns; `.detach().clone()` is the recommended form.
- `DynamicBatchSampler.__iter__` inherits an upstream PyG quirk: `skip_too_big`
  does not advance `num_processed`, so an oversized sample can be re-examined.
- No test suite. `test.py`, `test.ipynb`, `test2.ipynb` are scratch/benchmark
  files despite the pytest config in `pyproject.toml`.

## [v1.0.0]

- Repo setup

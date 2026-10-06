import importlib
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import pytest
import torch
from torch_geometric.data import Batch, Data

from step.models import DenoiseModel

SNAPSHOT = Path(__file__).parent / "snapshots" / "baseline.pt"

# Model configurations under test: constructor defaults (original architecture, random-walk PE),
# train.py defaults (sequence PE) and the rotation-invariant model with edge-length features.
CONFIGS = {
    "legacy_rw": dict(pe="rw"),
    "seq": dict(pe="seq"),
    "invariant": dict(pe="seq", invariant=True, edge_dim=8),
}
WALK_LENGTH = 8


def make_graph(n: int, seed: int) -> Data:
    torch.manual_seed(seed)
    x = torch.randint(0, 20, (n,))
    edge_index = torch.tensor(
        [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)],
        dtype=torch.long,
    ).T
    return Data(
        x=x,
        pos=torch.rand(n, 3) * 10,
        pe=torch.rand(n, WALK_LENGTH),
        edge_index=edge_index,
        noise=torch.rand(n, 3),
        mask=torch.rand(n) < 0.3,
        orig_x=x.clone(),
    )


def make_batch(seed: int = 0) -> Batch:
    return Batch.from_data_list([make_graph(20, seed), make_graph(13, seed + 1)])


def make_model(config: str, **kwargs) -> DenoiseModel:
    torch.manual_seed(0)
    return DenoiseModel(
        hidden_dim=64, pe_dim=4, pos_dim=8, num_layers=2, heads=2, walk_length=WALK_LENGTH, **CONFIGS[config], **kwargs
    )


def run_training(model: DenoiseModel, seed: int = 0, steps: int = 20) -> tuple[list[float], Batch]:
    """Run AdamW steps; return (losses, forward output of the last step).

    forward() clones the batch, so capture the step output with a hook.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    outputs = []
    forward = model.forward

    def capture(batch):
        out = forward(batch)
        outputs.append(out)
        return out

    model.forward = capture
    model.train()
    try:
        for _ in range(steps):
            batch = make_batch(seed)
            opt.zero_grad()
            loss = model.training_step(batch, 0)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    finally:
        model.forward = forward
    return losses, outputs[-1]


def test_import_everything():
    for name in [
        "step",
        "step.data",
        "step.data.datamodules",
        "step.data.datasets",
        "step.data.parsers",
        "step.data.samplers",
        "step.data.transforms",
        "step.data.utils",
        "step.models",
        "step.models.denoise",
        "step.models.downstream",
        "step.models.utils",
        "step.utils",
        "step.utils.tracking",
        "step.utils.cli",
        "step.utils.optim",
    ]:
        importlib.import_module(name)
    importlib.import_module("torch_geometric")
    importlib.import_module("lightning.pytorch")


@pytest.mark.parametrize("config", CONFIGS)
def test_forward_pass(config):
    batch = make_batch()
    model = make_model(config, predict_all=True)
    model.eval()
    inp = batch
    with torch.no_grad():
        out = model.forward(batch)
    total = out.x.shape[0]
    assert out.type_pred.shape == (total, 20)
    assert out.noise_pred.shape == (total, 3)
    # BUG-2 guard: forward must not mutate the input batch (x stays long ids).
    assert inp.x.dtype == torch.long
    assert "type_pred" not in inp


@pytest.mark.parametrize("config", CONFIGS)
def test_predict_step(config):
    """predict_step produces the frozen per-graph embeddings consumed by finetune.py."""
    batch = make_batch()
    model = make_model(config)
    model.eval()
    with torch.no_grad():
        out = model.predict_step(batch, 0)
    assert out.aggr_x.shape[0] == batch.num_graphs
    assert torch.isfinite(out.aggr_x).all()


@pytest.mark.parametrize("config", CONFIGS)
def test_training_step(config):
    batch = make_batch()
    model = make_model(config)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss).item()
    assert loss.item() < 10.0


@pytest.mark.parametrize("config", CONFIGS)
def test_loss_decreases(config):
    model = make_model(config)
    losses, _ = run_training(model)
    assert all(torch.isfinite(torch.tensor(loss)).item() for loss in losses)
    assert losses[-1] < losses[0]


def test_invariant_model_rotation():
    """Invariant model: embeddings are rotation-invariant and noise predictions rotate with the input."""
    model = make_model("invariant", attn_type="multihead")
    model.eval()
    batch = make_batch()
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(0)))
    rotated = batch.clone()
    rotated.pos = batch.pos @ q.T
    with torch.no_grad():
        out = model.forward(batch)
        out_rot = model.forward(rotated)
    assert torch.allclose(out.x, out_rot.x, atol=1e-4)
    assert torch.allclose(out.noise_pred @ q.T, out_rot.noise_pred, atol=1e-4)


def test_constructor_validation():
    with pytest.raises(ValueError):
        DenoiseModel(hidden_dim=64, pe_dim=32, pos_dim=32)
    with pytest.raises(ValueError):
        DenoiseModel(invariant=True, edge_dim=0)
    with pytest.raises(ValueError):
        DenoiseModel(pe="seq", pe_dim=63)


@pytest.fixture(scope="session")
def snapshot_data():
    snap = {}
    for config in CONFIGS:
        losses, batch = run_training(make_model(config), seed=1234)
        snap[config] = {
            "final_loss": float(losses[-1]),
            "noise_pred_mean": float(batch.noise_pred.detach().mean()),
            "type_pred_mean": float(batch.type_pred.detach().mean()),
        }
    if SNAPSHOT.exists():
        ref = torch.load(SNAPSHOT, weights_only=True)
        for config in snap:
            for k in snap[config]:
                assert torch.allclose(
                    torch.tensor(snap[config][k]), torch.tensor(ref[config][k]), rtol=1e-4, atol=1e-4
                ), f"{config}/{k}"
        return snap
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snap, SNAPSHOT)
    return snap


def test_seed_snapshot(snapshot_data):
    for config in CONFIGS:
        assert snapshot_data[config]["final_loss"] > 0


def test_transforms_compose():
    """Transforms must be instantiable and compose (pyg BaseTransform.forward abstract)."""
    from torch_geometric.transforms import Compose

    from step.data.transforms import MaskType, MaskTypeAnkh, MaskTypeBERT, PosNoise

    torch.manual_seed(0)
    n = 40
    x = torch.randint(0, 20, (n,))
    edges = torch.tensor([[0, 1], [1, 0]])
    out = Compose([PosNoise(0.5), MaskType(0.15)])(Data(x=x.clone(), pos=torch.rand(n, 3), edge_index=edges))
    assert out.noise.shape == (n, 3)
    for transform in (MaskType(0.15), MaskTypeAnkh(0.15), MaskTypeBERT(0.15, mask_prob=1.0, mut_prob=0.0)):
        out = transform(Data(x=x.clone(), pos=torch.rand(n, 3), edge_index=edges))
        name = type(transform).__name__
        assert out.mask.dtype == torch.bool and out.mask.shape == (n,), name
        assert out.mask.any(), name
        assert torch.equal(out.orig_x, x), name  # full-length ground truth
        assert set(out.x[out.mask].tolist()) == {20}, name
        assert torch.equal(out.x[~out.mask], x[~out.mask]), name


def test_mask_type_ankh():
    """MaskTypeAnkh: exact quota, one node per class first, deterministic under torch.manual_seed."""
    from step.data.transforms import MaskTypeAnkh

    def run(seed, x):
        torch.manual_seed(seed)
        return MaskTypeAnkh(0.15)(Data(x=x.clone()))

    torch.manual_seed(0)
    x = torch.randint(0, 20, (200,))
    a, b = run(1, x), run(1, x)
    assert torch.equal(a.mask, b.mask)
    assert int(a.mask.sum()) == 30
    assert set(a.orig_x[a.mask].tolist()) == set(x.tolist())
    # Fewer residue types than the quota: the rest is filled from unmasked nodes.
    few = run(0, torch.randint(0, 5, (100,)))
    assert int(few.mask.sum()) == 15


def test_masks_survive_batching():
    """Boolean masks and full-length orig_x stay aligned with nodes after collation."""
    from step.data.transforms import MaskTypeAnkh, MaskTypeBERT

    torch.manual_seed(0)
    for transform in (MaskTypeAnkh(0.2), MaskTypeBERT(0.2)):
        graphs = [transform(Data(x=torch.randint(0, 20, (n,)))) for n in (30, 40)]
        batch = Batch.from_data_list(graphs)
        assert torch.equal(batch.mask, torch.cat([g.mask for g in graphs]))
        assert torch.equal(batch.orig_x, torch.cat([g.orig_x for g in graphs]))
        assert batch.orig_x.shape[0] == batch.num_nodes


def test_graph_transforms():
    """graph_transforms builds radius edges, plus RandomWalkPE only for pe == "rw"."""
    from torch_geometric.transforms import Compose

    from step.data.transforms import graph_transforms

    torch.manual_seed(0)
    data = Data(x=torch.randint(0, 20, (50,)), pos=torch.rand(50, 3) * 30)
    rw = Compose(graph_transforms(10, "rw", walk_length=WALK_LENGTH))(data.clone())
    assert rw.edge_index.shape[1] > 0
    assert rw.pe.shape == (50, WALK_LENGTH)
    seq = Compose(graph_transforms(10, "seq"))(data.clone())
    assert "pe" not in seq
    assert torch.equal(seq.edge_index, rw.edge_index)


def test_classification_step():
    """Classification head must reduce the node dim before CE/metrics (BUG-6)."""
    from step.models.downstream import ClassificationModel

    torch.manual_seed(0)
    graphs = [Data(x=torch.randn(8), y=cls) for cls in (0, 1, 2, 3)]
    batch = Batch.from_data_list(graphs)
    model = ClassificationModel(num_classes=5, hidden_dim=8, dropout=0.0)
    model.train()
    out = model.shared_step(batch, 0, step_name="train")
    loss = out["loss"]
    assert torch.isfinite(loss).item()
    assert loss.item() < 10.0
    epoch = model.stage_metrics("train").compute()
    for key in ("train/acc", "train/auc", "train/mcc"):
        assert torch.isfinite(epoch[key]).item(), key


def test_downstream_metrics_are_epoch_level():
    """Correlations/AUROC are computed over the whole split, not averaged over batches (splits can be sorted)."""
    from scipy.stats import spearmanr
    from torchmetrics.functional import auroc

    from step.models.downstream import ClassificationModel, HomologyModel, RegressionModel

    torch.manual_seed(0)
    # Regression split sorted by target, in batches: within-batch Spearman is ~0, the dataset-level one is high
    y = torch.sort(torch.randn(64)).values
    model = RegressionModel(hidden_dim=8, dropout=0.0).eval()
    preds = []
    for chunk in y.split(16):
        batch = Batch.from_data_list([Data(x=torch.randn(4), y=v) for v in chunk])
        with torch.no_grad():
            model.linear(torch.zeros(1, 4))  # materialise the lazy layer
        # make the "prediction" the target plus noise, independent of the model
        p = chunk + 0.5 * torch.randn_like(chunk)
        model._log_epoch_metrics("test", p, chunk, len(chunk))
        preds.append(p)
    epoch = model.stage_metrics("test").compute()
    assert abs(epoch["test/spearman"].item() - spearmanr(torch.cat(preds), y).statistic) < 1e-5

    # Classification split sorted by label: one class per batch would make per-batch AUROC meaningless
    clf = ClassificationModel(num_classes=3, hidden_dim=8, dropout=0.0)
    labels = torch.arange(3).repeat_interleave(10)
    logits = torch.randn(30, 3) + 2 * torch.nn.functional.one_hot(labels, 3)
    for lg, lb in zip(logits.split(10), labels.split(10), strict=True):
        clf._log_epoch_metrics("val", lg, lb, len(lb))
    epoch = clf.stage_metrics("val").compute()
    assert torch.isclose(epoch["val/auc"], auroc(logits, labels, "multiclass", num_classes=3, average="weighted"))
    assert torch.isclose(epoch["val/acc"], (logits.argmax(1) == labels).float().mean())

    # Homology keeps separate metrics per test set
    homology = HomologyModel(num_classes=3)
    for stage in ("train", "val", "test_fold", "test_superfamily", "test_family"):
        assert homology.stage_metrics(stage).prefix == f"{stage}/"


def test_mlflow_logging(tmp_path, monkeypatch):
    """A short fit logs the model hyperparameters and step losses to the MLflow store."""
    import lightning.pytorch as pl
    import mlflow
    from torch_geometric.loader import DataLoader

    from step.utils import mlflow_logger

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    logger = mlflow_logger("smoke")
    model = make_model("seq")
    loader = DataLoader([make_graph(n, seed) for seed, n in enumerate((20, 13, 17, 9))], batch_size=2)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=2,
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["smoke"])
    assert len(runs) == 1
    run = runs.iloc[0]
    assert run["status"] == "FINISHED"
    assert run["params.pe"] == "seq" and run["params.hidden_dim"] == "64"
    assert run["metrics.train/loss_step"] > 0


def test_downstream_dataset_missing_raw_files(tmp_path, monkeypatch):
    """Downstream datasets are never downloaded; a missing raw file gives an actionable error."""
    from step.data.datasets import FluorescenceDataset

    monkeypatch.setattr(FluorescenceDataset, "root", str(tmp_path / "fluorescence"))
    with pytest.raises(FileNotFoundError, match="fluorescence_train.json") as err:
        FluorescenceDataset("train")
    assert "uvx wandb artifact get rindti/fluorescence/fluorescence_dataset:latest" in str(err.value)


M_JANNASCHII = Path(__file__).parents[1] / "data" / "m_jannaschii" / "raw" / "m_jannaschii"


@pytest.mark.skipif(not M_JANNASCHII.exists(), reason="needs the m_jannaschii foldcomp database in data/")
def test_deeploc_dataset(tmp_path, monkeypatch):
    """DeepLocDataset builds one graph per json record, from structures stored as foldcomp bytes in one HDF5 file."""
    import json

    import foldcomp
    import h5py
    import numpy as np

    from step.data.datasets import DeepLocDataset
    from step.data.parsers import ProtStructure
    from step.data.utils import extract_uniprot_id

    raw = tmp_path / "deeploc" / "raw"
    raw.mkdir(parents=True)
    expected = {}
    with foldcomp.open(str(M_JANNASCHII), decompress=False) as db, h5py.File(raw / "deeploc_structures.h5", "w") as h5:
        for i in range(8):
            fcz = db[i]
            title, pdb = foldcomp.decompress(fcz)
            acc = extract_uniprot_id(title)
            h5.create_dataset(acc, data=np.frombuffer(fcz, dtype=np.uint8))
            expected[acc] = ProtStructure(pdb)  # the decompress + PDB parse path the other datasets use
    accs = list(expected)
    split_accs = {"train": accs[:4], "valid": accs[4:6], "test": accs[6:]}
    for split, ids in split_accs.items():
        records = [dict(id=acc, label=(3 * i) % 10, location="x") for i, acc in enumerate(ids)]
        (raw / f"deeploc_{split}.json").write_text(json.dumps(records))

    monkeypatch.setattr(DeepLocDataset, "root", str(tmp_path / "deeploc"))
    for split, name in (("train", "train"), ("val", "valid"), ("test", "test")):
        ds = DeepLocDataset(split)
        assert len(ds) == len(split_accs[name])
        for i, graph in enumerate(ds):
            assert graph.id == split_accs[name][i]
            assert graph.y.dtype == torch.long and graph.y.numel() == 1
            assert int(graph.y) == (3 * i) % 10
            ref = Data(**expected[graph.id].get_graph())
            assert graph.seq == expected[graph.id].get_sequence() and len(graph.seq) == graph.num_nodes
            assert graph.x.dtype == torch.long and torch.equal(graph.x, ref.x)
            assert graph.pos.dtype == torch.float32 and torch.allclose(graph.pos, ref.pos, atol=1e-3)


def test_deeploc_missing_raw_files(tmp_path, monkeypatch):
    """DeepLoc's error points at the build script, not at W&B."""
    from step.data.datasets import DeepLocDataset

    monkeypatch.setattr(DeepLocDataset, "root", str(tmp_path / "deeploc"))
    with pytest.raises(FileNotFoundError, match="scripts/build_deeploc.py"):
        DeepLocDataset("train")


def test_deeploc_split_matching():
    """PEER records are matched to DeepLoc entries by sequence, through PEER's truncation of long sequences."""
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "build_deeploc.py"
    spec = importlib.util.spec_from_file_location("build_deeploc", path)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)

    long_seq = "M" + "A" * 600 + "C" * 600 + "W"
    deeploc = [
        dict(id="P1", location="Nucleus", membrane="U", test=False, sequence="MKV"),
        dict(id="P2", location="Cytoplasm-Nucleus", membrane="S", test=False, sequence=long_seq),
        dict(id="P3", location="Nucleus", membrane="U", test=False, sequence="MKV"),  # duplicate sequence
        dict(id="P4", location="Plastid", membrane="M", test=True, sequence="MLL"),
    ]
    peer = {
        "train": [("MKV", 6), (long_seq[:500] + long_seq[-500:], 1)],
        "valid": [("MKV", 6)],
        "test": [("MLL", 8)],
    }
    splits = build.match_splits(peer, deeploc)
    assert [r["id"] for r in splits["train"]] == ["P1", "P2"]
    assert [r["id"] for r in splits["valid"]] == ["P3"]
    assert splits["train"][1]["location"] == "Cytoplasm"
    assert splits["test"][0] == dict(id="P4", label=8, location="Plastid", membrane="M", sequence="MLL")
    with pytest.raises(AssertionError):  # a label that disagrees with DeepLoc's location
        build.match_splits({"train": [("MKV", 0)], "valid": [], "test": []}, deeploc[:1])


class ToyDownstreamDataset(list):
    """Stand-in for a downstream InMemoryDataset: a few random 'proteins' per split, transforms applied eagerly."""

    def __init__(self, split: str, transform=None, pre_transform=None, **kwargs):
        sizes = {"train": 12, "val": 4, "test": 5}[split]
        gen = torch.Generator().manual_seed(len(split))
        graphs = []
        for i in range(sizes):
            n = 8 + i % 5
            g = Data(x=torch.randint(0, 20, (n,), generator=gen), pos=torch.rand(n, 3, generator=gen) * 8, y=i % 3)
            for t in (pre_transform, transform):
                g = t(g) if t is not None else g
            graphs.append(g)
        super().__init__(graphs)


def save_toy_checkpoint(path: Path, config: str = "invariant") -> Path:
    """A Lightning-loadable checkpoint of a small DenoiseModel (radius 6, so the toy graphs have edges)."""
    import lightning

    model = make_model(config, radius=6)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        path,
    )
    return path


def toy_datamodule(ckpt: Path, **kwargs):
    """A DownstreamDataModule over ToyDownstreamDataset, embedding with the checkpoint at ``ckpt``."""
    from step.data.datamodules import DownstreamDataModule

    class ToyDataModule(DownstreamDataModule):
        dataset_class = ToyDownstreamDataset

    return ToyDataModule(
        feature_extract_model=str(ckpt),
        feature_extract_model_source="checkpoint",
        num_workers=0,
        batch_size=4,
        **kwargs,
    )


def test_make_head():
    """--head picks an MLP or a linear probe."""
    from step.models.downstream import ClassificationModel, make_head

    assert isinstance(make_head("linear", 8, 3, 0.0), torch.nn.LazyLinear)
    assert isinstance(ClassificationModel(num_classes=3, head="linear").linear, torch.nn.LazyLinear)
    with pytest.raises(ValueError):
        make_head("transformer", 8, 3, 0.0)


@pytest.mark.parametrize("config", ["invariant", "legacy_rw"])
def test_random_init_control(tmp_path, config):
    """The random-init encoder is seeded on its own, BatchNorm-calibrated, and shared by the fit and test stages."""
    ckpt = save_toy_checkpoint(tmp_path / "toy.ckpt", config)
    dms = []
    for global_seed in (1, 2):
        torch.manual_seed(global_seed)  # stands in for the head seed
        dm = toy_datamodule(ckpt, random_init=True, random_init_seed=7, bn_calib_batches=2)
        dm.setup("fit")
        dms.append(dm)
    a, b = (dict(dm.model.state_dict()) for dm in dms)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    pretrained = torch.load(ckpt, weights_only=False)["state_dict"]
    weight = next(k for k in pretrained if k.endswith("weight") and pretrained[k].dim() == 2)
    assert not torch.equal(a[weight], pretrained[weight])
    bns = [m for m in dms[0].model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert bns and all(not torch.allclose(bn.running_var, torch.ones_like(bn.running_var)) for bn in bns)
    assert all(bn.num_batches_tracked == 2 and bn.momentum == 0.1 and not bn.training for bn in bns)
    model = dms[0].model
    dms[0].setup("test")
    assert dms[0].model is model  # the test split is embedded by the same network as train


def test_embedding_standardization(tmp_path):
    """Embeddings are z-scored with the train split's statistics, and the cache holds them unstandardized."""
    ckpt = save_toy_checkpoint(tmp_path / "toy.ckpt")
    cache = tmp_path / "cache"
    dm = toy_datamodule(ckpt, embed_cache=str(cache))
    dm.setup("fit")
    dm.setup("test")
    train = torch.stack([d.x for d in dm.train])
    assert torch.allclose(train.mean(0), torch.zeros(train.shape[1]), atol=1e-5)
    assert torch.allclose(train.std(0), torch.ones(train.shape[1]), atol=1e-4)
    mean, std = dm.embed_stats
    raw_test = torch.stack([d.x for d in torch.load(cache / "test.pt", weights_only=False)])
    assert torch.allclose(torch.stack([d.x for d in dm.test]), (raw_test - mean) / std, atol=1e-5)
    # a test-only setup still standardizes with train statistics
    test_only = toy_datamodule(ckpt, embed_cache=str(cache))
    test_only.setup("test")
    assert torch.allclose(torch.stack([d.x for d in test_only.test]), torch.stack([d.x for d in dm.test]), atol=1e-5)
    raw = toy_datamodule(ckpt, standardize=False)
    raw.setup("test")
    assert torch.allclose(torch.stack([d.x for d in raw.test]), raw_test, atol=1e-5)


def test_embed_cache_key(tmp_path):
    """Pretrained caches without a key stay valid; random-init caches from another setup are recomputed."""
    ckpt = save_toy_checkpoint(tmp_path / "toy.ckpt")
    cache = tmp_path / "cache"
    cache.mkdir()
    sentinel = [Data(x=torch.full((4,), 3.0), y=0)]
    for split in ("train", "val", "test"):
        torch.save(sentinel, cache / f"{split}.pt")  # cache written before keys existed
    dm = toy_datamodule(ckpt, embed_cache=str(cache), standardize=False)
    dm.setup("fit")
    assert torch.equal(dm.train[0].x, sentinel[0].x)

    ri = toy_datamodule(ckpt, embed_cache=str(cache), standardize=False, random_init=True, bn_calib_batches=1)
    ri.setup("fit")
    assert len(ri.train) == 12 and not (cache / "test.pt").exists()  # stale test split dropped too
    ri.setup("test")
    first = torch.load(cache / "test.pt", weights_only=False)
    reseeded = toy_datamodule(
        ckpt, embed_cache=str(cache), standardize=False, random_init=True, random_init_seed=1, bn_calib_batches=1
    )
    reseeded.setup("test")
    assert not torch.allclose(reseeded.test[0].x, first[0].x)

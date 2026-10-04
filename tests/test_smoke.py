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
    for key in ("acc", "auc", "mcc"):
        assert torch.isfinite(out[key]).item(), key


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

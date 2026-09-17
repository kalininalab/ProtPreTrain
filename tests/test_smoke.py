import importlib
import os
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")

import pytest
import torch
from torch_geometric.data import Batch, Data

from step.models import DenoiseModel

SNAPSHOT = Path(__file__).parent / "snapshots" / "baseline.pt"


def make_graph(n: int, seed: int) -> Data:
    torch.manual_seed(seed)
    x = torch.randint(0, 20, (n,))
    edge_index = torch.tensor(
        [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)],
        dtype=torch.long,
    ).T
    return Data(
        x=x,
        pos=torch.rand(n, 3),
        pe=torch.rand(n, 8),
        edge_index=edge_index,
        noise=torch.rand(n, 3),
        mask=torch.rand(n) < 0.3,
        orig_x=x.clone(),
    )


def make_batch(seed: int = 0) -> Batch:
    return Batch.from_data_list([make_graph(20, seed), make_graph(13, seed + 1)])


def make_model() -> DenoiseModel:
    torch.manual_seed(0)
    return DenoiseModel(hidden_dim=64, pe_dim=4, pos_dim=8, num_layers=2, heads=2, walk_length=8)


def run_training(model: DenoiseModel, seed: int = 0, steps: int = 20) -> tuple[list[float], Batch]:
    """Run AdamW steps; return (losses, forward output of the last step).

    forward() now clones the batch, so capture the step output with a hook to keep
    the exact same forward/optimizer/BatchNorm update count as the original run.
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
            losses.append(float(loss))
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
        "step.data.transforms",
        "step.data.utils",
        "step.models",
        "step.models.denoise",
        "step.models.downstream",
        "step.models.utils",
        "step.utils",
        "step.utils.checkpoint",
        "step.utils.cli",
        "step.utils.math",
        "step.utils.optim",
        "step.utils.vis",
    ]:
        importlib.import_module(name)
    importlib.import_module("torch_geometric")
    importlib.import_module("pytorch_lightning")


def test_forward_pass():
    batch = make_batch()
    model = make_model()
    model.eval()
    with torch.no_grad():
        batch = model.forward(batch)
    total = batch.x.shape[0]
    assert batch.type_pred.shape == (total, 20)
    assert batch.noise_pred.shape == (total, 3)


def test_training_step():
    batch = make_batch()
    model = make_model()
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss).item()
    assert loss.item() < 5.0


def test_loss_decreases():
    batch = make_batch()
    model = make_model()
    losses, _ = run_training(model)
    assert all(torch.isfinite(torch.tensor(loss)).item() for loss in losses)
    assert losses[-1] < losses[0]


@pytest.fixture(scope="session")
def snapshot_data():
    model = make_model()
    losses, batch = run_training(model, seed=1234)
    snap = {
        "final_loss": float(losses[-1]),
        "noise_pred_mean": float(batch.noise_pred.detach().mean()),
        "type_pred_mean": float(batch.type_pred.detach().mean()),
    }
    if SNAPSHOT.exists():
        ref = torch.load(SNAPSHOT, weights_only=True)
        for k in snap:
            assert torch.allclose(torch.tensor(snap[k]), torch.tensor(ref[k]), rtol=1e-4, atol=1e-4), k
        return snap
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snap, SNAPSHOT)
    return snap


def test_seed_snapshot(snapshot_data):
    assert snapshot_data["final_loss"] > 0


def test_transforms_compose():
    """Transforms must be instantiable and compose (pyg BaseTransform.forward abstract)."""
    from torch_geometric.transforms import Compose

    from step.data.transforms import MaskType, MaskTypeAnkh, PosNoise

    torch.manual_seed(0)
    n = 20
    x = torch.randint(0, 20, (n,))
    x_ref = x.clone()
    edges = torch.tensor([[0, 1], [1, 0]])
    data = Data(x=x, pos=torch.rand(n, 3), edge_index=edges)
    out = Compose([PosNoise(0.5), MaskType(0.15)])(data)
    assert out.mask.dtype == torch.bool
    assert torch.equal(out.orig_x, x_ref)
    assert set(out.x[out.mask].tolist()).issubset({20})

    # MaskTypeAnkh overwrites mask with an index tensor; masked nodes are set to 20.
    ankh = MaskTypeAnkh(0.15)
    out2 = ankh(Data(x=x.clone(), pos=torch.rand(n, 3), edge_index=edges))
    assert out2.mask.numel() > 0
    assert set(out2.x[out2.mask].tolist()).issubset({20})


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

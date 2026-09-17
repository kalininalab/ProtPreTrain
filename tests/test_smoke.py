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
    """training_step mutates batch.x in place, so rebuild a fresh batch each step."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    model.train()
    for _ in range(steps):
        batch = make_batch(seed)
        opt.zero_grad()
        loss = model.training_step(batch, 0)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses, batch


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
        "step.utils.checkpoint",
        "step.utils.cli",
        "step.utils.optim",
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

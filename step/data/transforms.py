import random

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj


class RandomWalkPE(BaseTransform):
    """Random walk dense version."""

    def __init__(self, walk_length: int, attr_name: str = "pe", cuda: bool = True):
        self.walk_length = walk_length
        self.attr_name = attr_name
        self.cuda = cuda

    def forward(self, data: Data) -> Data:
        if self.cuda:
            if torch.cuda.device_count() > 1:
                device = random.randint(0, torch.cuda.device_count() - 1)
                data = data.to(f"cuda:{device}")
            else:
                data = data.to("cuda")
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0)).squeeze(0)
        row_sums = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sums.clamp(min=1)
        pe_list = [None] * self.walk_length
        pe_list[0] = torch.zeros(adj.size(0))
        walk_matrix = adj
        for i in range(1, self.walk_length):
            walk_matrix = walk_matrix @ adj
            pe_list[i] = walk_matrix.diag()
        pe = torch.stack(pe_list, dim=-1)
        data[self.attr_name] = pe
        return data.to("cpu")


class PosNoise(BaseTransform):
    """Add Gaussian noise to the coordinates of the nodes in a graph."""

    def __init__(self, sigma: float = 0.5, plddt_dependent: bool = False):
        self.sigma = sigma
        self.plddt_dependent = plddt_dependent

    def forward(self, batch) -> torch.Tensor:
        noise = torch.randn_like(batch.pos) * self.sigma
        if self.plddt_dependent:
            noise *= 2 - batch.plddt.unsqueeze(-1) / 100
        batch.pos += noise
        batch.noise = noise
        return batch


class MaskType(BaseTransform):
    """Masks the type of the nodes in a graph."""

    def __init__(self, pick_prob: float):
        self.prob = pick_prob

    def forward(self, batch) -> torch.Tensor:
        mask = torch.rand_like(batch.x, dtype=torch.float32) < self.prob
        batch.orig_x = batch.x.clone()
        batch.x[mask] = 20
        batch.mask = mask
        return batch


class MaskTypeAnkh(BaseTransform):
    """Masks a ``pick_prob`` fraction of nodes, preferring one node per amino-acid class.

    Class order is shuffled and all random draws use the torch RNG, so masking is
    deterministic under a fixed ``torch.manual_seed`` (the previous implementation
    used Python ``random`` and a ``set``, which were not torch-seed controlled).
    """

    def __init__(self, pick_prob: float):
        self.prob = pick_prob

    def forward(self, batch) -> torch.Tensor:
        N = batch.x.size(0)
        n = int(N * self.prob)
        batch.orig_x = batch.x.clone()

        # One node per class, classes visited in shuffled order.
        picked = []
        for cls in torch.randperm(20):
            if len(picked) >= n:
                break
            idx = (batch.x == cls).nonzero(as_tuple=False).flatten()
            if idx.numel() > 0:
                picked.append(idx[torch.randint(idx.numel(), ())].item())

        # Fill any remaining quota uniformly from the unmasked nodes.
        if len(picked) < n:
            remaining = torch.tensor([i for i in range(N) if i not in set(picked)])
            perm = torch.randperm(remaining.numel())
            picked.extend(remaining[perm[: n - len(picked)]].tolist())

        mask = torch.tensor(picked, dtype=torch.long)
        batch.x[mask] = 20
        batch.mask = mask
        return batch


class MaskTypeBERT(BaseTransform):
    """Masks the type of the nodes in a graph with BERT-like system."""

    def __init__(self, pick_prob: float, mask_prob: float = 0.8, mut_prob: float = 0.1):
        self.pick_prob = pick_prob
        self.mask_prob = mask_prob
        self.mut_prob = mut_prob

    def forward(self, batch) -> torch.Tensor:
        n = batch.x.size(0)
        num_changed_nodes = int(n * self.pick_prob)  # 0.15 in BERT paper
        num_masked_nodes = int(num_changed_nodes * self.mask_prob)  # 0.8 in BERT paper
        num_mutated_nodes = int(num_changed_nodes * self.mut_prob)  # 0.1 in BERT paper
        indices = torch.randperm(n)[:num_changed_nodes]  # All nodes that are changed in some way
        batch.orig_x = batch.x[indices].clone()
        batch.mask = indices
        mask_indices = indices[:num_masked_nodes]  # All nodes that are masked
        mut_indices = indices[num_masked_nodes : num_masked_nodes + num_mutated_nodes]  # All nodes that are mutated
        batch.x[mask_indices] = 20
        batch.x[mut_indices] = torch.randint_like(batch.x[mut_indices], low=0, high=20)
        return batch


class SequenceOnly:
    """Removes all node features except the sequence."""

    def __call__(self, batch) -> torch.Tensor:
        n = batch.x.size(0)
        batch.pos = torch.stack(
            [torch.arange(0, n) * 3.8 - (3.8 * (n - 1) / 2), torch.zeros(n), torch.zeros(n)], dim=1
        )
        return batch


class StructureOnly(MaskType):
    """Mask everything"""

    def __init__(self):
        super().__init__(pick_prob=1.0)

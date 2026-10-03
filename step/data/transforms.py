import random

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj


def graph_transforms(radius: float, pe: str = "rw", walk_length: int = 20, cuda: bool = False) -> list:
    """Transforms that build the residue graph from `pos`: radius edges, plus RandomWalkPE when pe == "rw".

    The "seq" and "none" encodings are computed inside the model, so they need no transform.
    """
    transforms = [T.RadiusGraph(radius), T.ToUndirected()]
    if pe == "rw":
        transforms.append(RandomWalkPE(walk_length, attr_name="pe", cuda=cuda))
    return transforms


class RandomWalkPE(BaseTransform):
    """Random walk dense version."""

    def __init__(self, walk_length: int, attr_name: str = "pe", cuda: bool = True):
        self.walk_length = walk_length
        self.attr_name = attr_name
        self.cuda = cuda

    def forward(self, data: Data) -> Data:
        if self.cuda and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                device = random.randint(0, torch.cuda.device_count() - 1)
                data = data.to(f"cuda:{device}")
            else:
                data = data.to("cuda")
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0)).squeeze(0)
        row_sums = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sums.clamp(min=1)
        pe_list = [None] * self.walk_length
        pe_list[0] = adj.new_zeros(adj.size(0))
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
    deterministic under a fixed ``torch.manual_seed``.
    """

    def __init__(self, pick_prob: float):
        self.prob = pick_prob

    def forward(self, batch) -> torch.Tensor:
        N = batch.x.size(0)
        n = int(N * self.prob)
        # Store as a boolean mask, not an index tensor: PyG concatenates custom
        # attributes without adding node offsets, so per-graph indices silently
        # point into the wrong graph once a batch is collated.
        bool_mask = torch.zeros(N, dtype=torch.bool)
        # One node per class, classes visited in shuffled order.
        num_picked = 0
        for cls in torch.randperm(20):
            if num_picked >= n:
                break
            idx = (batch.x == cls).nonzero(as_tuple=False).flatten()
            if idx.numel() > 0:
                bool_mask[idx[torch.randint(idx.numel(), ())]] = True
                num_picked += 1
        # Fill any remaining quota uniformly from the unmasked nodes.
        if num_picked < n:
            remaining = (~bool_mask).nonzero(as_tuple=False).flatten()
            bool_mask[remaining[torch.randperm(remaining.numel())[: n - num_picked]]] = True
        batch.orig_x = batch.x.clone()
        batch.x[bool_mask] = 20
        batch.mask = bool_mask
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
        # orig_x must stay full-length: training_step indexes it as orig_x[mask]
        # (predict_all=False) or compares it against per-node logits (predict_all=True).
        batch.orig_x = batch.x.clone()
        bool_mask = torch.zeros(n, dtype=torch.bool)
        bool_mask[indices] = True
        batch.mask = bool_mask
        mask_indices = indices[:num_masked_nodes]  # All nodes that are masked
        mut_indices = indices[num_masked_nodes : num_masked_nodes + num_mutated_nodes]  # All nodes that are mutated
        batch.x[mask_indices] = 20
        batch.x[mut_indices] = torch.randint_like(batch.x[mut_indices], low=0, high=20)
        return batch


class SequenceOnly:
    """Replace coordinates with a straight line (3.8 A spacing), so only sequence order is left as structure.

    Must run before any structure-derived pre-transform (RadiusGraph, RandomWalkPE, ...), otherwise edges and PE
    still come from the true structure.
    """

    def __call__(self, batch) -> torch.Tensor:
        n = batch.x.size(0)
        batch.pos = torch.stack(
            [torch.arange(0, n) * 3.8 - (3.8 * (n - 1) / 2), torch.zeros(n), torch.zeros(n)], dim=1
        )
        return batch


class StructureOnly(MaskType):
    """Mask residue types with probability pick_prob (1.0 masks everything)."""

    def __init__(self, pick_prob: float = 1.0):
        super().__init__(pick_prob=pick_prob)

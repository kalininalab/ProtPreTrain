import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class PosNoise(BaseTransform):
    """Add Gaussian noise to the coordinates of the nodes in a graph."""

    def __init__(self, sigma: float = 0.5, plddt_dependent: bool = False):
        self.sigma = sigma
        self.plddt_dependent = plddt_dependent

    def __call__(self, batch) -> torch.Tensor:
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

    def __call__(self, batch) -> torch.Tensor:
        mask = torch.rand_like(batch.x, dtype=torch.float32) < self.prob
        batch.orig_x = batch.x[mask].clone()
        batch.x[mask] = 20
        batch.mask = mask
        return batch


class MaskTypeBERT(BaseTransform):
    """Masks the type of the nodes in a graph with BERT-like system."""

    def __init__(self, pick_prob: float, mask_prob: float = 0.8, mut_prob: float = 0.1):
        self.pick_prob = pick_prob
        self.mask_prob = mask_prob
        self.mut_prob = mut_prob

    def __call__(self, batch) -> torch.Tensor:
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


class MaskTypeWeighted(MaskType):
    """Masks the type of the nodes in a graph."""

    def __call__(self, batch) -> torch.Tensor:
        num_mut = int(batch.x.size(0) * self.prob)
        num_mut_per_aa = int(num_mut / 20)
        mask = []
        for i in range(20):
            indices = torch.where(batch.x == i)[0]
            random_pick = torch.randperm(indices.size(0))[:num_mut_per_aa]
            mask.append(indices[random_pick])
        mask = torch.cat(mask)
        batch.orig_x = batch.x[mask].clone()
        batch.x[mask] = 20
        batch.mask = mask
        return batch

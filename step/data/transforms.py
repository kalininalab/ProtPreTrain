import random

import scipy
import torch
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
        batch.orig_x = batch.x.clone()
        batch.x[mask] = 20
        batch.mask = mask
        return batch


class MaskTypeAnkh(BaseTransform):
    """Ensures each amino acid is masked at least once in a graph."""

    def __init__(self, pick_prob: float):
        self.prob = pick_prob

    def __call__(self, batch) -> torch.Tensor:
        N = batch.x.size(0)
        n = int(N * self.prob)
        mask = set()
        aas = torch.randperm(20)
        for i in aas:
            if len(mask) >= n:
                break
            subset = torch.where(batch.x == i)[0]
            if subset.size(0) > 0:
                mask.add(subset[random.randint(0, subset.size(0) - 1)].item())
        if n < 20:
            mask = torch.tensor(list(mask))
        else:
            all_indices = set(range(N))
            remaining_indices = list(all_indices - mask)
            random.shuffle(remaining_indices)
            mask = list(mask) + remaining_indices[: n - len(mask)]
            mask = torch.tensor(list(mask))
        batch.orig_x = batch.x.clone()
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


class SequenceOnly(BaseTransform):
    """Removes all node features except the sequence."""

    def __call__(self, batch) -> torch.Tensor:
        n = batch.x.size(0)

        # delete all edges between non-neighboring nodes
        batch.orig_edge_index = batch.edge_index.clone()
        tmp = torch.abs(batch.edge_index[0, :] - batch.edge_index[1, :])
        batch.edge_index = torch.tensor(torch.stack(list(filter(lambda x: x <= 1, tmp))), dtype=torch.long)

        # arrange nodes as a line on the x-axis to remove positional-structural information
        # inside torch.arange, I adjusted the numbers to center the atoms at (0, 0, 0). The original formula is
        # torch.arange(0, n * 3.8, 3.8) - (n-1) * 1.9, which can be reformulated to the following:
        batch.pos = torch.stack((torch.arange(-(n-1) * 1.9, (n + 1) * 1.9, 3.8), torch.zeros(n), torch.zeros(n)), dim=1)

        return batch


class StructureOnly(MaskType):
    def __init__(self):
        super().__init__(pick_prob=1.0)


class SphericalHarmonics(BaseTransform):
    def __init__(self, degree):
        self.degree = degree

    def spherical_harmonics(self, coords, degree):
        r, theta, phi = self.cartesian_to_spherical(coords[:, 0], coords[:, 1], coords[:, 2])

        harmonics = []
        for m in range(-degree, degree + 1):
            Y_m_l = [
                scipy.special.sph_harm(m, degree, phi_i.item(), theta_i.item()) for phi_i, theta_i in zip(phi, theta)
            ]
            Y_m_l = torch.tensor(Y_m_l, dtype=torch.complex64)
            harmonics.append(Y_m_l)

        return torch.stack(harmonics, dim=-1)

    @staticmethod
    def cartesian_to_spherical(x, y, z):
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.acos(z / r)  # polar angle
        phi = torch.atan2(y, x)  # azimuthal angle
        return r, theta, phi

    def __call__(self, data):
        coords = data.pos
        harmonics = self.spherical_harmonics(coords, self.degree)
        data.sph_harmonics = harmonics
        return data

from typing import List

import torch_geometric.transforms as T
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from .datasets import FoldSeekDataset, FoldSeekSmallDataset, FluorescenceDataset, StabilityDataset
from .samplers import DynamicBatchSampler


class FoldSeekDataModule(LightningDataModule):
    """Base data module, contains all the datasets for train, val and test."""

    def __init__(
        self,
        transforms: List[BaseTransform] = [],
        pre_transforms: List[BaseTransform] = [],
        batch_size: int = 128,
        num_workers: int = 1,
        shuffle: bool = True,
        batch_sampling: bool = False,
        max_num_nodes: int = 0,
    ):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.pre_transforms = pre_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.batch_sampling = batch_sampling
        self.max_num_nodes = max_num_nodes

    def _get_dataloader(self, ds: Dataset) -> DataLoader:
        if self.batch_sampling:
            assert self.max_num_nodes > 0
            sampler = DynamicBatchSampler(ds, self.max_num_nodes, self.shuffle)
            return DataLoader(ds, batch_sampler=sampler, num_workers=self.num_workers)
        else:
            return DataLoader(ds, **self._dl_kwargs(False))

    def train_dataloader(self):
        """Train dataloader."""
        return self._get_dataloader(self.train)

    def setup(self, stage: str = None):
        """Load the individual datasets."""
        pre_transform = T.Compose(self.pre_transforms)
        transform = T.Compose(self.transforms)
        self.train = FoldSeekDataset(
            transform=transform,
            pre_transform=pre_transform,
        )

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
        )


class FoldSeekSmallDataModule(FoldSeekDataModule):
    def setup(self, stage: str = None):
        """Load the individual datasets."""
        pre_transform = T.Compose(self.pre_transforms)
        transform = T.Compose(self.transforms)
        self.train = FoldSeekSmallDataset(
            root=self.root,
            transform=transform,
            pre_transform=pre_transform,
        )


class DownstreamDataModule(LightningDataModule):
    dataset_class = None
    root = None

    def __init__(
        self,
        transforms: List[BaseTransform] = [],
        pre_transforms: List[BaseTransform] = [],
        batch_size: int = 128,
        num_workers: int = 1,
        shuffle: bool = True,
        batch_sampling: bool = False,
        max_num_nodes: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.transforms = transforms
        self.pre_transforms = pre_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.batch_sampling = batch_sampling
        self.max_num_nodes = max_num_nodes
        self.kwargs = kwargs

    def _get_dataloader(self, ds: Dataset) -> DataLoader:
        if self.batch_sampling:
            assert self.max_num_nodes > 0
            sampler = DynamicBatchSampler(ds, self.max_num_nodes, self.shuffle)
            return DataLoader(ds, batch_sampler=sampler, num_workers=self.num_workers)
        else:
            return DataLoader(ds, **self._dl_kwargs(False))

    def train_dataloader(self):
        """Train dataloader."""
        return self._get_dataloader(self.train)

    def val_dataloader(self):
        """Validation dataloader."""
        return self._get_dataloader(self.val)

    def test_dataloader(self):
        """Test dataloader."""
        return self._get_dataloader(self.test)

    def setup(self, stage: str = None):
        """Load the individual datasets."""
        pre_transform = T.Compose(self.pre_transforms)
        transform = T.Compose(self.transforms)
        self.train = self.dataset_class(
            transform=transform,
            pre_transform=pre_transform,
            **self.kwargs,
        )
        self.val = self.dataset_class(
            transform=transform,
            pre_transform=pre_transform,
            **self.kwargs,
        )
        self.test = self.dataset_class(
            transform=transform,
            pre_transform=pre_transform,
            **self.kwargs,
        )

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
        )


class FluorescenceDataModule(DownstreamDataModule):
    dataset_class = FluorescenceDataset


class StabilityDataModule(DownstreamDataModule):
    dataset_class = StabilityDataset

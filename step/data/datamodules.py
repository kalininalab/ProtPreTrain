from pathlib import Path
from typing import List

import torch
import torch_geometric.transforms as T
from pytorch_lightning import LightningDataModule, Trainer
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

import wandb

from ..models import DenoiseModel
from .datasets import (
    FluorescenceDataset,
    FluorescenceESMDataset,
    FoldSeekDataset,
    FoldSeekSmallDataset,
    StabilityDataset,
    StabilityESMDataset,
)
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
    """Small subset of foldseek for debugging."""

    def setup(self, stage: str = None):
        """Load the individual datasets."""
        pre_transform = T.Compose(self.pre_transforms)
        transform = T.Compose(self.transforms)
        self.train = FoldSeekSmallDataset(
            transform=transform,
            pre_transform=pre_transform,
        )


class DownstreamDataModule(LightningDataModule):
    """Abstract class for downstream tasks."""

    dataset_class = None

    def __init__(
        self,
        feature_extract_model: str,
        transforms: List[BaseTransform] = [],
        pre_transforms: List[BaseTransform] = [],
        batch_size: int = 128,
        num_workers: int = 8,
        shuffle: bool = True,
        batch_sampling: bool = False,
        max_num_nodes: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.feature_extract_model = feature_extract_model
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

    def load_pretrained_model(self):
        """Load the pretrained model from wandb."""
        artifact = wandb.run.use_artifact(self.feature_extract_model, type="model")
        artifact_dir = artifact.download()
        model = DenoiseModel.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
        model.eval()
        return model

    def setup(self, stage: str = None):
        """Load the individual datasets."""
        pre_transform = T.Compose(self.pre_transforms)
        transform = T.Compose(self.transforms)
        self.train = self.dataset_class(
            "train",
            transform=transform,
            pre_transform=pre_transform,
            **self.kwargs,
        )
        self.val = self.dataset_class(
            "val",
            transform=transform,
            pre_transform=pre_transform,
            **self.kwargs,
        )
        self.test = self.dataset_class(
            "test",
            transform=transform,
            pre_transform=pre_transform,
            **self.kwargs,
        )
        trainer = Trainer(callbacks=[], logger=False, accelerator="gpu", precision="bf16-mixed")
        model = self.load_pretrained_model()
        for split in ["train", "val", "test"]:
            dl = self._get_dataloader(getattr(self, split))
            result = trainer.predict(model, dl)
            data_list = []
            for batch in result:
                for i in range(len(batch)):
                    data = Data(x=batch.x[i], y=batch.y[i])
                    data_list.append(data)
            if split == "train":
                self.train = data_list
            elif split == "val":
                self.val = data_list
            elif split == "test":
                self.test = data_list

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
        )


class FluorescenceDataModule(DownstreamDataModule):
    """Predict fluorescence change."""

    dataset_class = FluorescenceDataset


class StabilityDataModule(DownstreamDataModule):
    """Predict peptide stability."""

    dataset_class = StabilityDataset

from pathlib import Path
from typing import List, Literal

import ankh
import torch
import torch_geometric.transforms as T
from pytorch_lightning import LightningDataModule, Trainer
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
from transformers import pipeline

import wandb

from ..models import DenoiseModel
from .datasets import FluorescenceDataset, FoldSeekDataset, FoldSeekDatasetSmall, HomologyDataset, StabilityDataset
from .samplers import DynamicBatchSampler
from .transforms import RandomWalkPE, SequenceOnly, StructureOnly


class FoldSeekDataModule(LightningDataModule):
    """Base data module, contains all the datasets for train, val and test."""

    dataset_class = FoldSeekDataset

    def __init__(
        self,
        transforms: List[BaseTransform] = [],
        pre_transforms: List[BaseTransform] = [],
        batch_size: int = 128,
        num_workers: int = 1,
        shuffle: bool = True,
        batch_sampling: bool = False,
        max_num_nodes: int = 0,
        subset: int = None,
    ):
        super().__init__()
        self.transforms = transforms
        self.pre_transforms = pre_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.batch_sampling = batch_sampling
        self.max_num_nodes = max_num_nodes
        self.subset = subset

    def _get_dataloader(self, ds: Dataset) -> DataLoader:
        if self.batch_sampling:
            assert self.max_num_nodes > 0
            if self.shuffle:
                sampler = RandomSampler(ds)
            else:
                sampler = SequentialSampler(ds)
            batch_sampler = DynamicBatchSampler(
                ds,
                sampler,
                mode="node",
                max_num=self.max_num_nodes,
                skip_too_big=True,
                num_steps=len(ds),
            )
            return DataLoader(
                ds,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            return DataLoader(ds, **self._dl_kwargs(False))

    def train_dataloader(self):
        """Train dataloader."""
        return self._get_dataloader(self.train)

    def setup(self, stage: str = None):
        """Load the individual datasets."""
        pre_transform = T.Compose(self.pre_transforms)
        transform = T.Compose(self.transforms)
        self.train = self.dataset_class(
            transform=transform,
            pre_transform=pre_transform,
            num_workers=self.num_workers,
        )
        if self.subset:
            self.train = self.train[: self.subset]

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class FoldSeekDataModuleSmall(FoldSeekDataModule):
    dataset_class = FoldSeekDatasetSmall


class DownstreamDataModule(LightningDataModule):
    """Abstract class for downstream tasks."""

    dataset_class = None

    def __init__(
        self,
        feature_extract_model: str,
        feature_extract_model_source: str,
        batch_size: int = 128,
        num_workers: int = 8,
        shuffle: bool = True,
        ablation: Literal["none", "sequence", "structure"] = "none",
        radius: int = 10,
        walk_length: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.feature_extract_model = feature_extract_model
        self.feature_extract_model_source = feature_extract_model_source
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.ablation = ablation
        self.radius = radius
        self.walk_length = walk_length
        self.kwargs = kwargs

    def _optional_add_transform(self):
        if self.feature_extract_model_source == "wandb":
            pre_transform = T.Compose([T.Center(), T.NormalizeRotation()])
            transform = [
                T.RadiusGraph(self.radius),
                T.ToUndirected(),
                RandomWalkPE(self.walk_length, "pe"),
            ]
            if self.ablation == "sequence":
                transform = [SequenceOnly()] + transform
            elif self.ablation == "structure":
                transform = [StructureOnly()] + transform
            transform = T.Compose(transform)
        else:
            pre_transform = None
            transform = None
        return transform, pre_transform

    def _get_dataloader(self, ds: Dataset) -> DataLoader:
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

    def _load_wandb_model(self):
        artifact = wandb.run.use_artifact(self.feature_extract_model, type="model")
        artifact_dir = artifact.download()
        p = Path(artifact_dir)
        p = [x for x in p.glob("*.ckpt")][0]
        model = DenoiseModel.load_from_checkpoint(p)
        model.eval()
        return model

    def load_pretrained_model(self):
        """Load the pretrained model."""
        if self.feature_extract_model_source == "wandb":
            return self._load_wandb_model()
        elif self.feature_extract_model_source == "huggingface":
            return pipeline(
                "feature-extraction",
                model=self.feature_extract_model,
                device=0,
            )
        elif self.feature_extract_model_source == "ankh":
            if self.feature_extract_model == "ankh-base":
                model, tokenizer = ankh.load_base_model()
            elif self.feature_extract_model == "ankh-large":
                model, tokenizer = ankh.load_large_model()
            model.eval()
            return model, tokenizer
        else:
            raise ValueError(f"Unknown feature extract model source {self.feature_extract_model_source}")

    def setup(self, stage: str = None):
        """Load the individual datasets."""
        transform, pre_transform = self._optional_add_transform()
        splits = []
        if stage == "fit" or stage is None:
            splits.append("train")
            splits.append("val")
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
        if stage == "test" or stage is None:
            splits.append("test")
            self.test = self.dataset_class(
                "test",
                transform=transform,
                pre_transform=pre_transform,
                **self.kwargs,
            )
        if self.feature_extract_model_source == "wandb":
            self._embed_with_wandb(splits)
        elif self.feature_extract_model_source == "huggingface":
            self._embed_with_huggingface(splits)
        elif self.feature_extract_model_source == "ankh":
            self._embed_with_ankh(splits)
        else:
            raise ValueError(f"Unknown feature extract model source {self.feature_extract_model_source}")

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
        )

    def _assign_data(self, split: str, data_list: List[Data]):
        if split == "train":
            self.train = data_list
        elif split == "val":
            self.val = data_list
        elif split == "test":
            self.test = data_list

    def _embed_with_wandb(self, splits: List[str]) -> List[Data]:
        trainer = Trainer(
            callbacks=[],
            logger=False,
            accelerator="gpu",
            precision="bf16-mixed",
        )
        model = self.load_pretrained_model()
        for split in splits:
            dl = self._get_dataloader(getattr(self, split))
            result = trainer.predict(model, dl)
            data_list = []
            for batch in result:
                for i in range(len(batch)):
                    data = Data(x=batch.aggr_x[i], y=batch.y[i])
                    data_list.append(data)
            self._assign_data(split, data_list)

    def _embed_with_huggingface(self, splits: List[str]) -> List[Data]:
        pipe = self.load_pretrained_model()
        for split in splits:
            print(split)
            ds = getattr(self, split)
            data_list = []
            for i in tqdm(ds):
                x = torch.tensor(pipe(i.seq)).squeeze(0).mean(dim=0)
                y = i.y
                data = Data(x=x, y=y)
                data_list.append(data)
            self._assign_data(split, data_list)

    def _embed_with_ankh(self, splits: List[str]) -> List[Data]:
        model, tokenizer = self.load_pretrained_model()
        model.to("cuda")
        for split in splits:
            print(split)
            ds = getattr(self, split)
            data_list = []
            for i in tqdm(ds):
                outputs = tokenizer.batch_encode_plus(
                    [list(i.seq)],
                    add_special_tokens=True,
                    padding=True,
                    is_split_into_words=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    embeddings = model(
                        input_ids=outputs["input_ids"].to("cuda"),
                        attention_mask=outputs["attention_mask"].to("cuda"),
                    )
                x = embeddings["last_hidden_state"][0].squeeze(0).mean(dim=0).cpu()
                y = i.y
                data = Data(x=x, y=y)
                data_list.append(data)
            self._assign_data(split, data_list)


class FluorescenceDataModule(DownstreamDataModule):
    """Predict fluorescence change."""

    dataset_class = FluorescenceDataset


class StabilityDataModule(DownstreamDataModule):
    """Predict peptide stability."""

    dataset_class = StabilityDataset


class HomologyDataModule(DownstreamDataModule):
    """Predict remote homology."""

    dataset_class = HomologyDataset

    def setup(self, stage: str = None):
        """Load the individual datasets."""
        transform, pre_transform = self._optional_add_transform()
        splits = []
        if stage == "fit" or stage is None:
            splits.append("train")
            splits.append("val")
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
        if stage == "test" or stage is None:
            splits += ["test_fold", "test_family", "test_superfamily"]
            self.test_fold = self.dataset_class(
                "test_fold",
                transform=transform,
                pre_transform=pre_transform,
                **self.kwargs,
            )
            self.test_superfamily = self.dataset_class(
                "test_superfamily",
                transform=transform,
                pre_transform=pre_transform,
                **self.kwargs,
            )
            self.test_family = self.dataset_class(
                "test_family",
                transform=transform,
                pre_transform=pre_transform,
                **self.kwargs,
            )
        if self.feature_extract_model_source == "wandb":
            self._embed_with_wandb(splits)
        elif self.feature_extract_model_source == "huggingface":
            self._embed_with_huggingface(splits)
        elif self.feature_extract_model_source == "ankh":
            self._embed_with_ankh(splits)
        else:
            raise ValueError(f"Unknown feature extract model source {self.feature_extract_model_source}")

    def test_dataloader(self):
        """Test dataloader."""
        return [
            self._get_dataloader(self.test_fold),
            self._get_dataloader(self.test_superfamily),
            self._get_dataloader(self.test_family),
        ]

    def _assign_data(self, split: str, data_list: List[Data]):
        if split == "train":
            self.train = data_list
        elif split == "val":
            self.val = data_list
        elif split == "test_fold":
            self.test_fold = data_list
        elif split == "test_superfamily":
            self.test_superfamily = data_list
        elif split == "test_family":
            self.test_family = data_list

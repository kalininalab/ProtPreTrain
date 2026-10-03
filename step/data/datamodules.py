import os
import warnings
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch
import torch_geometric.transforms as T
import wandb
from lightning.pytorch import LightningDataModule, Trainer
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from ..models import DenoiseModel
from .datasets import DTIDataset, FluorescenceDataset, FoldCompDataset, HomologyDataset, StabilityDataset
from .samplers import DynamicBatchSampler
from .transforms import SequenceOnly, StructureOnly, graph_transforms


class FoldCompDataModule(LightningDataModule):
    """Base data module, contains all the datasets for train, val and test."""

    def __init__(
        self,
        db_name: str = "afdb_rep_v4",
        transforms: Optional[List[BaseTransform]] = None,
        pre_transforms: Optional[List[BaseTransform]] = None,
        batch_size: int = 128,
        num_workers: int = 1,
        shuffle: bool = True,
        batch_sampling: bool = False,
        max_num_nodes: int = 0,
        subset: int = None,
        max_length: int = None,
        val_size: int = 0,
    ):
        super().__init__()
        self.db_name = db_name
        self.transforms = list(transforms or [])
        self.pre_transforms = list(pre_transforms or [])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.batch_sampling = batch_sampling
        self.max_num_nodes = max_num_nodes
        self.subset = subset
        self.max_length = max_length
        self.val_size = val_size
        self.val = None

    def _get_dataloader(self, ds: Dataset, shuffle: bool = False) -> DataLoader:
        if self.batch_sampling:
            batch_sampler = DynamicBatchSampler(self.lengths, self.max_num_nodes, shuffle=self.shuffle)
            return DataLoader(ds, batch_sampler=batch_sampler, num_workers=self.num_workers, pin_memory=True)
        return DataLoader(ds, **self._dl_kwargs(shuffle))

    def train_dataloader(self):
        """Train dataloader."""
        return self._get_dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        """Held-out structures, in fixed-size batches (only when val_size > 0)."""
        if self.val is None:
            return []
        return DataLoader(self.val, **self._dl_kwargs(shuffle=False))

    def _dataset(self) -> FoldCompDataset:
        return FoldCompDataset(
            db_name=self.db_name,
            transform=T.Compose(self.transforms),
            pre_transform=T.Compose(self.pre_transforms),
            num_workers=self.num_workers,
        )

    def prepare_data(self):
        """Process the dataset and cache lengths on a single process, so DDP ranks don't race on first use."""
        ds = self._dataset()
        if self.max_length or self.batch_sampling:
            ds.lengths()

    def setup(self, stage: str = None):
        """Load the individual datasets."""
        self.train = self._dataset()
        # lengths stay aligned with self.train, the batch sampler sizes batches from them
        keep = np.arange(len(self.train))
        if self.max_length or self.batch_sampling:
            self.lengths = self.train.lengths()
            if self.max_length:
                keep = np.flatnonzero(self.lengths <= self.max_length)
        keep = keep[: self.subset]
        # The val split is a fixed random draw, independent of the training seed
        keep = np.random.default_rng(0).permutation(keep) if self.val_size else keep
        val, keep = keep[: self.val_size], keep[self.val_size :]
        full = self.train
        self.train = full.index_select(keep.tolist())
        self.val = full.index_select(val.tolist()) if self.val_size else None
        if self.batch_sampling:
            self.lengths = self.lengths[keep]

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


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
        ablation_maskfrac: float = 1.0,
        random_init: bool = False,
        embed_cache: str = None,
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
        self.ablation_maskfrac = ablation_maskfrac
        self.random_init = random_init
        self.embed_cache = embed_cache
        self.radius = radius
        self.walk_length = walk_length
        self.kwargs = kwargs
        if ablation == "sequence" and feature_extract_model_source in ("wandb", "checkpoint"):
            # Sequence ablation changes the pre_transform, so it needs its own processed files
            self.kwargs["processed_name"] = "processed_sequence"

    @property
    def _uses_denoise_model(self) -> bool:
        return self.feature_extract_model_source in ("wandb", "checkpoint")

    def _optional_add_transform(self, hparams: dict):
        if self._uses_denoise_model:
            # SequenceOnly goes first, so that edges and PE are built from the straight-line positions
            pre_transform = [SequenceOnly()] if self.ablation == "sequence" else []
            pre_transform = T.Compose(pre_transform + [T.Center(), T.NormalizeRotation()])
            # The graph is built at embedding time from the encoder's own hyperparameters, so processed files
            # don't go stale when radius or PE type change between checkpoints
            transform = [StructureOnly(self.ablation_maskfrac)] if self.ablation == "structure" else []
            transform += graph_transforms(
                hparams.get("radius", self.radius),
                hparams.get("pe", "rw"),
                hparams.get("walk_length", self.walk_length),
            )
            transform = T.Compose(transform)
        else:
            pre_transform = None
            transform = None
        return transform, pre_transform

    def _get_dataloader(self, ds: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(ds, **self._dl_kwargs(shuffle))

    def train_dataloader(self):
        """Train dataloader."""
        return self._get_dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        """Validation dataloader."""
        return self._get_dataloader(self.val)

    def test_dataloader(self):
        """Test dataloader."""
        return self._get_dataloader(self.test)

    def _load_denoise_model(self):
        if self.feature_extract_model_source == "checkpoint":
            p = self.feature_extract_model
        else:
            artifact = wandb.run.use_artifact(self.feature_extract_model, type="model")
            artifact_dir = artifact.download()
            p = [x for x in Path(artifact_dir).glob("*.ckpt")][0]
        model = DenoiseModel.load_from_checkpoint(p, map_location="cpu")
        if self.random_init:
            # No-pretraining control: same architecture and hyperparameters, fresh weights
            model = DenoiseModel(**model.hparams)
        model.eval()
        return model

    def load_pretrained_model(self):
        """Load the pretrained model."""
        if self._uses_denoise_model:
            return self._load_denoise_model()
        elif self.feature_extract_model_source == "huggingface":
            from transformers import pipeline

            return pipeline(
                "feature-extraction",
                model=self.feature_extract_model,
                device=0,
            )
        elif self.feature_extract_model_source == "ankh":
            import ankh

            if self.feature_extract_model == "ankh-base":
                model, tokenizer = ankh.load_base_model()
            elif self.feature_extract_model == "ankh-large":
                model, tokenizer = ankh.load_large_model()
            model.eval()
            return model, tokenizer
        elif self.feature_extract_model_source == "prostt5":
            """
            Mainly based on
            https://github.com/mheinzinger/ProstT5/blob/bfc140799e3aed6d0e2f9e0e8965a8746f2dbbc2/scripts/embed.py#L20
            """
            from transformers import T5EncoderModel, T5Tokenizer

            model = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
            model = model.eval().half()
            vocab = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
            return model, vocab
        else:
            raise ValueError(f"Unknown feature extract model source {self.feature_extract_model_source}")

    def _load_model_and_transforms(self):
        """Load our encoder (if used) first, since its hyperparameters decide how the graphs are built."""
        self.model = self.load_pretrained_model() if self._uses_denoise_model else None
        return self._optional_add_transform(dict(self.model.hparams) if self.model else {})

    def setup(self, stage: str = None):
        """Load the individual datasets."""
        transform, pre_transform = self._load_model_and_transforms()
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
        self.embed_splits(splits)

    def embed_splits(self, splits: List[str]):
        """Embed the splits, reusing per-split embeddings from `embed_cache` when present."""
        cached = {}
        if self.embed_cache:
            os.makedirs(self.embed_cache, exist_ok=True)
            for split in splits:
                path = os.path.join(self.embed_cache, f"{split}.pt")
                if os.path.exists(path):
                    cached[split] = torch.load(path, weights_only=False)
        todo = [split for split in splits if split not in cached]
        if todo:
            self._embed(todo)
            if self.embed_cache:
                for split in todo:
                    torch.save(getattr(self, split), os.path.join(self.embed_cache, f"{split}.pt"))
        for split, data_list in cached.items():
            self._assign_data(split, data_list)

    def _embed(self, splits: List[str]):
        if self._uses_denoise_model:
            self._embed_with_denoise_model(splits)
        elif self.feature_extract_model_source == "huggingface":
            self._embed_with_huggingface(splits)
        elif self.feature_extract_model_source == "ankh":
            self._embed_with_ankh(splits)
        elif self.feature_extract_model_source == "prostt5":
            self._embed_with_prostt5(splits)
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

    def _embed_with_denoise_model(self, splits: List[str]) -> List[Data]:
        trainer = Trainer(
            callbacks=[],
            logger=False,
            accelerator="auto",
            # CPUs without native bf16 run bf16 autocast several times slower than fp32
            precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        )
        for split in splits:
            dl = self._get_dataloader(getattr(self, split))
            result = trainer.predict(self.model, dataloaders=dl)
            data_list = []
            for batch in result:
                for i in range(len(batch)):
                    k = batch[i]
                    k.x = batch.aggr_x[i]
                    data_list.append(k)
            self._assign_data(split, data_list)

    def _embed_with_huggingface(self, splits: List[str]) -> List[Data]:
        pipe = self.load_pretrained_model()
        for split in splits:
            print(split)
            ds = getattr(self, split)
            data_list = []
            for i in tqdm(ds):
                i.x = torch.tensor(pipe(i.seq)).squeeze(0).mean(dim=0)
                data_list.append(i)
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
                i.x = embeddings["last_hidden_state"][0].squeeze(0).mean(dim=0).cpu()
                data_list.append(i)
            self._assign_data(split, data_list)

    def _embed_with_prostt5(self, splits: List[str]) -> List[Data]:
        """
        Mainly based on
        https://github.com/mheinzinger/ProstT5/blob/bfc140799e3aed6d0e2f9e0e8965a8746f2dbbc2/scripts/embed.py#L54
        Interesting are lines 63, 89-91, 101-129
        """

        model, vocab = self.load_pretrained_model()
        model.to("cuda")
        for split in splits:
            print(split)
            ds = getattr(self, split)
            data_list = []
            for i in tqdm(ds):
                seq = i.seq.replace("U", "X").replace("Z", "X").replace("O", "X")
                seq = " ".join(["<fold2AA>"] + list(seq))
                token_encoding = vocab.batch_encode_plus(
                    [seq], add_special_tokens=True, padding="longest", return_tensors="pt"
                ).to("cuda")
                try:
                    with torch.no_grad():
                        embedding_repr = model(token_encoding.input_ids, attention_mask=token_encoding.attention_mask)
                except torch.cuda.OutOfMemoryError:
                    warnings.warn(f"prostt5 embedding OOM for {i} (L={len(i.seq)}), skipping", stacklevel=2)
                    continue
                i.x = embedding_repr.last_hidden_state[0, 1 : len(i.seq) + 1].mean(dim=0)
                data_list.append(i)
            self._assign_data(split, data_list)


class FluorescenceDataModule(DownstreamDataModule):
    """Predict fluorescence change."""

    dataset_class = FluorescenceDataset


class StabilityDataModule(DownstreamDataModule):
    """Predict peptide stability."""

    dataset_class = StabilityDataset


class DTIDataModule(DownstreamDataModule):
    """Predict peptide stability."""

    dataset_class = DTIDataset


class HomologyDataModule(DownstreamDataModule):
    """Predict remote homology."""

    dataset_class = HomologyDataset

    def setup(self, stage: str = None):
        """Load the individual datasets."""
        transform, pre_transform = self._load_model_and_transforms()
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
        self.embed_splits(splits)

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

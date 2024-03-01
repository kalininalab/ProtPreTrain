import multiprocessing
import os
import shutil
import subprocess
import time
from math import floor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import foldcomp
import h5py
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data, Dataset, InMemoryDataset, extract_tar
from tqdm.auto import tqdm

import wandb

from .parsers import ProtStructure
from .utils import apply_edits, compute_edits, extract_uniprot_id, get_start_end, save_file, smiles_to_ecfp


class FoldCompDataset(Dataset):
    """Save FoldSeekDB as a PyTorch Geometric dataset, using the on-disk format."""

    def __init__(
        self,
        db_name: str = "afdb_rep_v4",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        num_workers: int = 16,
        chunk_size: int = 4096,
    ) -> None:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
        self.db_name = db_name
        self.pre_transform = pre_transform
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        super().__init__(root=f"data/{db_name}", transform=transform, pre_transform=pre_transform)

    @property
    def raw_file_names(self):
        """Files that have to be present in the raw directory, foldcomp database."""
        return [self.db_name + x for x in self._db_extensions]

    @property
    def processed_file_names(self):
        """Files that have to be present in the processed directory, skip some for speed."""
        return [f"data/chunk_{a}.h5" for a, _ in self._get_chunks()]

    @property
    def _db_extensions(self):
        """Extensions of the files that make up the foldcomp database."""
        return ["", ".dbtype", ".index", ".lookup", ".source"]

    def download(self):
        """Download the database using foldcomp.setup."""
        print("Downloading database...")
        current_dir = os.getcwd()
        os.chdir(self.raw_dir)
        foldcomp.setup(self.raw_file_names[0])
        os.chdir(current_dir)

    def process_chunk(self, start_num: int, end_num: int):
        """Process a single chunk of the database. This is done in parallel."""
        data_dict = {}
        with foldcomp.open(self.raw_paths[0]) as db:
            for idx in range(start_num, end_num):
                name, pdb = db[idx]
                ps = ProtStructure(pdb)
                data = Data.from_dict(ps.get_graph())
                data.uniprot_id = extract_uniprot_id(name)
                if self.pre_transform:
                    data = self.pre_transform(data)
                data_dict[idx] = data
        with h5py.File(self._chunk_name(start_num), "w") as h5py_file:
            for idx, data in data_dict.items():
                group = h5py_file.create_group(f"data_{idx}")
                for k, v in data.items():
                    group.create_dataset(k, data=v)

    def _get_chunks(self) -> List[tuple[int, int]]:
        with foldcomp.open(self.raw_paths[0]) as db:
            num_entries = len(db)
        l = [(x, x + self.chunk_size) for x in range(0, num_entries, self.chunk_size)]
        l[-1] = (l[-1][0], num_entries)
        return l

    def _chunk_name(self, start_num: int) -> str:
        return f"{self.processed_dir}/data/chunk_{start_num}.h5"

    def process(self) -> None:
        """Process the whole dataset for the dataset."""
        os.makedirs(f"{self.processed_dir}/data", exist_ok=True)
        print("Processing chunks in parallel...")
        Parallel(n_jobs=self.num_workers)(
            delayed(self.process_chunk)(start, finish) for start, finish in self._get_chunks()
        )

    def get(self, idx: int) -> Any:
        """Get a single datapoint from the dataset."""
        filename = self._chunk_name(idx // self.chunk_size * self.chunk_size)
        with h5py.File(filename, "r") as h5py_file:
            group = h5py_file[f"data_{idx}"]
            data = {}
            for k, v in group.items():
                if isinstance(v, h5py.Dataset):
                    if v.shape == ():
                        data[k] = str(v[()], "utf-8")
                    else:
                        data[k] = torch.from_numpy(v[:])
                else:
                    data[k] = v
            data = Data.from_dict(data)
            return data

    def len(self):
        with foldcomp.open(self.raw_paths[0]) as db:
            n = len(db)
        return n


class DownstreamDataset(InMemoryDataset):
    """Abstract class for downstream datasets. self._prepare_data should be implemented."""

    splits = {"train": 0, "val": 1, "test": 2}
    root = None
    wandb_name = None

    def __init__(self, split: str, *, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.splits[split]])

    def download(self):
        """Download the dataset from wandb."""
        if wandb.run is None:
            wandb.init()
        artifact = wandb.use_artifact(self.wandb_name, type="dataset")
        artifact_dir = artifact.download(self.raw_dir)
        extract_tar(str(Path(artifact_dir) / "dataset.tar.gz"), self.raw_dir)

    @property
    def processed_file_names(self):
        """Files that have to be present in the processed directory."""
        return ["train.pt", "valid.pt", "test.pt"]

    def _prepare_data(self, df: pd.DataFrame) -> List[Data]:
        raise NotImplementedError

    def process(self):
        """Do the full run for the dataset."""
        for split, idx in self.splits.items():
            df = pd.read_json(self.raw_paths[idx])
            data_list = self._prepare_data(df)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[idx])


class FluorescenceDataset(DownstreamDataset):
    """Predict fluorescence for GFP mutants."""

    root = "data/fluorescence"
    wandb_name = "rindti/fluorescence/fluorescence_dataset:latest"

    @property
    def raw_file_names(self):
        """Files that have to be present in the raw directory."""
        return [
            "fluorescence_train.json",
            "fluorescence_valid.json",
            "fluorescence_test.json",
            "AF-P42212-F1-model_v4.pdb",
        ]

    def _prepare_data(self, df: pd.DataFrame) -> List[Data]:
        # df["primary"] = "M" + df["primary"]
        ps = ProtStructure(self.raw_paths[3])
        orig_sequence = ps.get_sequence()
        orig_graph = Data(**ps.get_graph())
        df["edits"] = df.apply(lambda row: compute_edits(orig_sequence, row["primary"]), axis=1)
        df["graph"] = df.apply(lambda row: apply_edits(orig_graph, row["edits"]), axis=1)
        data_list = [
            Data(
                y=row["log_fluorescence"][0],
                num_mutations=row["num_mutations"],
                id=row["id"],
                seq=row["primary"],
                **row["graph"].to_dict(),
            )
            for _, row in df.iterrows()
        ]
        return data_list


class StabilityDataset(DownstreamDataset):
    """Predict stability for various proteins."""

    root = "data/stability"
    wandb_name = "rindti/stability/stability_dataset:latest"

    @property
    def raw_file_names(self):
        """Files that have to be present in the raw directory."""
        return [
            "stability_train.json",
            "stability_valid.json",
            "stability_test.json",
            "stability_db",
            "stability_db.index",
            "stability_db.lookup",
            "stability_db.dbtype",
        ]

    def _prepare_data(self, df: pd.DataFrame) -> List[Data]:
        data_list = []
        ids = df["id"].tolist()
        df.set_index("id", inplace=True)

        with foldcomp.open(self.raw_paths[3], ids=ids) as db:
            for name, pdb in tqdm(db):
                struct = ProtStructure(pdb)
                graph = Data(**struct.get_graph())
                graph["y"] = df.loc[name, "stability_score"][0]
                if isinstance(graph["y"], list):
                    graph["y"] = graph["y"][0]
                graph["seq"] = struct.get_sequence()
                data_list.append(graph)
        return data_list


class HomologyDataset(DownstreamDataset):
    splits = {"train": 0, "val": 1, "test_fold": 2, "test_superfamily": 3, "test_family": 4}
    root = "data/homology"
    wandb_name = "rindti/homology/homology_dataset:latest"

    @property
    def raw_file_names(self):
        """Files that have to be present in the raw directory."""
        return [
            "remote_homology_train.json",
            "remote_homology_valid.json",
            "remote_homology_test_fold_holdout.json",
            "remote_homology_test_superfamily_holdout.json",
            "remote_homology_test_family_holdout.json",
            "homology_db",
            "homology_db.index",
            "homology_db.lookup",
            "homology_db.dbtype",
        ]

    @property
    def processed_file_names(self):
        """Has some extra files for the test splits."""
        return ["train.pt", "valid.pt", "test_fold.pt", "test_superfamily.pt", "test_family.pt"]

    def _prepare_data(self, df: pd.DataFrame) -> List[Data]:
        data_list = []
        ids = df["id"].tolist()
        df.set_index("id", inplace=True)

        with foldcomp.open(self.raw_paths[5], ids=ids) as db:
            for name, pdb in tqdm(db):
                struct = ProtStructure(pdb)
                graph = Data(**struct.get_graph())
                graph["y"] = torch.tensor(df.loc[name, "fold_label"], dtype=torch.long)
                graph["seq"] = struct.get_sequence()
                data_list.append(graph)
        return data_list


class DTIDataset(DownstreamDataset):
    """LP-PDBBind, molecules as ECFP."""

    root = "data/dti"
    wandb_name = "rindti/dti/dti_dataset:latest"

    @property
    def raw_file_names(self):
        """Files that have to be present in the raw directory."""
        return [
            "dti_train.json",
            "dti_valid.json",
            "dti_test.json",
            "dti_db",
            "dti_db.index",
            "dti_db.lookup",
            "dti_db.dbtype",
        ]

    def _prepare_data(self, df: pd.DataFrame) -> List[Data]:
        data_list = []
        df.set_index("ids", inplace=True)
        ids = df.index.to_list()

        with foldcomp.open(self.raw_paths[3], ids=ids) as db:
            for name, pdb in tqdm(db):
                struct = ProtStructure(pdb)
                graph = Data(**struct.get_graph())
                graph["y"] = float(df.loc[name, "y"])
                graph["ecfp"] = smiles_to_ecfp(df.loc[name, "Ligand"], nbits=1024)
                graph["seq"] = struct.get_sequence()
                data_list.append(graph)
        return data_list

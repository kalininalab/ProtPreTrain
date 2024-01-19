import multiprocessing
import os
import shutil
import subprocess
import time
from math import floor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import foldcomp
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
        num_workers: int = 1,
    ) -> None:
        self.db_name = db_name
        self.pre_transform = pre_transform
        self.num_workers = num_workers
        super().__init__(root=f"data/{db_name}", transform=transform, pre_transform=pre_transform)

    @property
    def raw_file_names(self):
        """Files that have to be present in the raw directory, foldcomp database."""
        return [self.db_name + x for x in self._db_extensions]

    @property
    def processed_file_names(self):
        """Files that have to be present in the processed directory, skip some for speed."""
        return [f"data/data_{i}.pt" for i in range(0, len(self), 1000)]

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

    def process_chunk(self, start_num: int, end_num: int, chunk_id: int):
        """Process a single chunk of the database. This is done in parallel."""
        cpu_count = multiprocessing.cpu_count()
        torch.set_num_threads(floor(cpu_count / self.num_workers))
        with foldcomp.open(self.raw_paths[0]) as db:
            for idx in tqdm(range(start_num, end_num), smoothing=0.1, leave=True, position=chunk_id):
                name, pdb = db[idx]
                ps = ProtStructure(pdb)
                data = Data.from_dict(ps.get_graph())
                data.uniprot_id = extract_uniprot_id(name)
                if self.pre_transform:
                    data = self.pre_transform(data)
                torch.save(data, f"{self.processed_dir}/data/data_{idx}.pt")

    def process(self) -> None:
        """Process the whole dataset for the dataset."""
        os.makedirs(f"{self.processed_dir}/data", exist_ok=True)
        print("Processing chunks in parallel...")
        with foldcomp.open(self.raw_paths[0]) as db:
            num_entries = len(db)
        chunk_indices = get_start_end(num_entries, self.num_workers)
        Parallel(n_jobs=self.num_workers)(
            delayed(self.process_chunk)(start, finish, idx) for idx, (start, finish) in enumerate(chunk_indices)
        )
        self.clean()

    def get(self, idx: int) -> Any:
        """Get a single datapoint from the dataset."""
        return torch.load(f"{self.processed_dir}/data/data_{idx}.pt")

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
    wandb_name = "ilsenatorov/fluorescence/fluorescence_dataset:latest"

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
    wandb_name = "ilsenatorov/stability/stability_dataset:latest"

    @property
    def raw_file_names(self):
        """Files that have to be present in the raw directory."""
        return [
            "stability_train.json",
            "stability_valid.json",
            "stability_test.json",
            "structures_db",
            "structures_db.index",
            "structures_db.lookup",
            "structures_db.dbtype",
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
    wandb_name = "ilsenatorov/homology/homology_dataset:latest"

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
            "LP_PDBBind.csv",
            "dti_db",
            "dti_db.index",
            "dti_db.lookup",
            "dti_db.dbtype",
        ]

    def _prepare_data(self, df: pd.DataFrame) -> List[Data]:
        data_list = []
        df = pd.read_csv(self.raw_paths[0], index_col=0)
        ids = df.index.to_list()

        with foldcomp.open(self.raw_paths[3], ids=ids) as db:
            for name, row in tqdm(df.iterrows()):
                name, pdb = db.get(name)
                struct = ProtStructure(pdb)
                graph = Data(**struct.get_graph())
                graph["y"] = row["value"]
                graph["ecfp"] = smiles_to_ecfp(row["smiles"], nbits=1024)
                graph["seq"] = struct.get_sequence()
                data_list.append(graph)
        return data_list

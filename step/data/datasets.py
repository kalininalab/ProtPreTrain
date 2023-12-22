import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import foldcomp
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data, InMemoryDataset, OnDiskDataset, extract_tar
from tqdm.auto import tqdm

import wandb

from .parsers import ProtStructure
from .utils import apply_edits, compute_edits, extract_uniprot_id


class FoldSeekDataset(OnDiskDataset):
    """Save FoldSeekDB as a PyTorch Geometric dataset, using the on-disk format."""

    def __init__(
        self,
        root: str = "data/foldseek/",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        backend: str = "sqlite",
        num_workers: int = 1,
        chunk_size: int = 1024,
    ) -> None:
        self._pre_transform = pre_transform
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        schema = {
            "x": {"dtype": torch.int64, "size": (-1,)},
            "edge_index": {"dtype": torch.int64, "size": (2, -1)},
            "uniprot_id": str,
            "pe": {"dtype": torch.float32, "size": (-1, 20)},
            "pos": {"dtype": torch.float32, "size": (-1, 3)},
        }
        super().__init__(root, transform, backend=backend, schema=schema)

    @property
    def raw_file_names(self):
        return ["afdb_rep_v4" + x for x in self._db_extensions]

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

    def get_chunks(self):
        """Break the original fold database into chunks."""
        nchunks = self.num_workers
        subprocess.run(
            [
                "mmseqs",
                "splitdb",
                "--split",
                f"{nchunks}",
                self.raw_paths[0],
                f"{self.processed_dir}/chunks/chunk",
            ]
        )

    def process_chunk(self, chunk: int):
        """Process a single chunk of the database. This is done in parallel."""
        chunk_file = f"{self.processed_dir}/chunks/chunk_{chunk}_{self.num_workers}"
        with foldcomp.open(chunk_file) as db:
            data_list = []

            for idx, (name, pdb) in tqdm(
                enumerate(db),
                position=chunk,
                total=len(db),
                desc=f"Process {chunk}",
                smoothing=0.0,
                leave=False,
                mininterval=1.0,
            ):
                if len(ProtStructure(pdb)) > 1022:
                    continue
                data = Data(**ProtStructure(pdb).get_graph())
                data.uniprot_id = extract_uniprot_id(name)
                if self._pre_transform:
                    data = self._pre_transform(data)
                data_list.append(data.to_dict())

                if len(data_list) >= self.chunk_size:
                    torch.save(data_list, f"{self.processed_dir}/data/data_{chunk}_{idx}.pt")
                    data_list = []
            torch.save(data_list, f"{self.processed_dir}/data/data_{chunk}_{idx}.pt")

    def process(self) -> None:
        """Process the whole dataset for the dataset."""
        os.makedirs(f"{self.processed_dir}/chunks", exist_ok=True)
        os.makedirs(f"{self.processed_dir}/data", exist_ok=True)
        print("Chunking database...")
        self.get_chunks()
        print("Processing chunks in parallel...")
        Parallel(n_jobs=self.num_workers)(delayed(self.process_chunk)(chunk) for chunk in range(self.num_workers))
        print("Merging batches...")
        self.merge_batches()
        print("Cleaning up...")
        self.clean()

    def merge_batches(self):
        """Once all chunks are processed, merge them into a single database."""
        p = Path(self.processed_dir) / "data"
        batches_list = [x for x in p.glob("data*.pt")]
        for batch_file in tqdm(batches_list, total=len(batches_list), leave=True):
            data_list = torch.load(batch_file)
            self.extend(data_list)

    def clean(self):
        """Remove the temporary files."""
        p = Path(self.processed_dir)
        shutil.rmtree(p / "chunks")
        shutil.rmtree(p / "data")

    def serialize(self, data: Dict) -> Dict[str, Any]:
        """To dict method for the database."""
        return data

    def deserialize(self, data: Dict[str, Any]) -> Data:
        """From dict method for the database."""
        return Data.from_dict(data)


class FoldSeekDatasetSmall(FoldSeekDataset):
    """Small version of the FoldSeek dataset (just e_coli). Used for debugging."""

    def __init__(self, root: str = "data/foldseek_small/", *args, **kwargs) -> None:
        super().__init__(root=root, *args, **kwargs)

    @property
    def raw_file_names(self):
        return ["e_coli" + x for x in self._db_extensions]


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

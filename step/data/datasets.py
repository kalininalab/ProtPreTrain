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
from filelock import FileLock
from joblib import Parallel, delayed
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm.auto import tqdm

from .parsers import ProtStructure, aminoacids
from .utils import apply_edits, compute_edits, extract_uniprot_id, foldcomp_ca, smiles_to_ecfp


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
        self.db_name = db_name
        self.pre_transform = pre_transform
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        root = f"data/{db_name}"
        os.makedirs(root, exist_ok=True)
        # Concurrent jobs on a fresh database would download and process it simultaneously; one does, the rest wait
        with FileLock(f"{root}/.prepare.lock"):
            super().__init__(root=root, transform=transform, pre_transform=pre_transform)

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
        """Extensions of the files that make up the foldcomp database.

        ``.source`` is left out: foldcomp.setup fetches it when the server has one, but only afdb_rep_v4 does, so
        requiring it made every other database re-download on each load.
        """
        return ["", ".dbtype", ".index", ".lookup"]

    def download(self):
        """Download the database using foldcomp.setup."""
        print("Downloading database...")
        current_dir = os.getcwd()
        os.chdir(self.raw_dir)
        foldcomp.setup(self.raw_file_names[0])
        os.chdir(current_dir)

    def process_chunk(self, start_num: int, end_num: int):
        """Process a single chunk of the database. This is done in parallel."""
        # One thread per joblib worker. Not set in __init__, since that would also pin the training process
        torch.set_num_threads(1)
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
        chunks = [(x, x + self.chunk_size) for x in range(0, num_entries, self.chunk_size)]
        chunks[-1] = (chunks[-1][0], num_entries)
        return chunks

    def _chunk_name(self, start_num: int) -> str:
        return f"{self.processed_dir}/data/chunk_{start_num}.h5"

    def process(self) -> None:
        """Process the whole dataset for the dataset."""
        os.makedirs(f"{self.processed_dir}/data", exist_ok=True)
        print("Processing chunks in parallel...")
        Parallel(n_jobs=max(self.num_workers, 1))(
            delayed(self.process_chunk)(start, finish) for start, finish in self._get_chunks()
        )

    def _chunk_lengths(self, start_num: int, end_num: int) -> np.ndarray:
        with h5py.File(self._chunk_name(start_num), "r") as h5py_file:
            return np.array([h5py_file[f"data_{idx}"]["x"].shape[0] for idx in range(start_num, end_num)])

    def lengths(self) -> np.ndarray:
        """Number of residues per structure, read from HDF5 shapes and cached in the processed dir."""
        path = f"{self.processed_dir}/lengths.npy"
        with FileLock(f"{path}.lock"):
            if not os.path.exists(path):
                chunks = Parallel(n_jobs=max(self.num_workers, 1))(
                    delayed(self._chunk_lengths)(start, finish) for start, finish in self._get_chunks()
                )
                np.save(path, np.concatenate(chunks))
        return np.load(path)

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

    def __init__(
        self, split: str, *, transform=None, pre_transform=None, pre_filter=None, processed_name: str = "processed"
    ):
        self.processed_name = processed_name
        os.makedirs(self.root, exist_ok=True)
        # Parallel finetune jobs would otherwise process the same files at once; one does, the rest wait and load
        with FileLock(os.path.join(self.root, f".{processed_name}.lock")):
            super().__init__(self.root, transform, pre_transform, pre_filter)
        # weights_only=False is required: these archives hold pickled PyG Data
        # objects, and torch>=2.6 defaults weights_only=True, which refuses them.
        self.data, self.slices = torch.load(self.processed_paths[self.splits[split]], weights_only=False)

    def download(self):
        """Downstream datasets are not downloaded automatically: their raw files must already be in raw_dir."""
        task = Path(self.root).name
        missing = [f for f in self.raw_file_names if not os.path.exists(os.path.join(self.raw_dir, f))]
        raise FileNotFoundError(
            f"{type(self).__name__}: missing raw files in {self.raw_dir}: {', '.join(missing)}\n"
            f"Place them there by hand. The original copy is the W&B artifact rindti/{task}/{task}_dataset, e.g.:\n"
            f"  uvx wandb artifact get rindti/{task}/{task}_dataset:latest --root {self.raw_dir}\n"
            f"  tar -xzf {self.raw_dir}/dataset.tar.gz -C {self.raw_dir}"
        )

    @property
    def processed_dir(self) -> str:
        """Processed dir is configurable, so differently pre-transformed versions do not overwrite each other."""
        return os.path.join(self.root, self.processed_name)

    @property
    def processed_file_names(self):
        """Files that have to be present in the processed directory."""
        return ["train.pt", "valid.pt", "test.pt"]

    def _prepare_data(self, df: pd.DataFrame) -> List[Data]:
        raise NotImplementedError

    def process(self):
        """Do the full run for the dataset."""
        for idx in self.splits.values():
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


class DeepLocDataset(DownstreamDataset):
    """Subcellular localization, 10 classes (DeepLoc 1.0), on AlphaFold structures.

    Data: Almagro Armenteros et al., "DeepLoc: prediction of protein subcellular localization using deep learning",
    Bioinformatics 33(21):3387-3395 (2017), https://services.healthtech.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta
    (14,004 SwissProt proteins, 2,773 of them marked test).

    Split: PEER (Xu et al., "PEER: A Comprehensive and Multi-Task Benchmark for Protein Sequence Understanding",
    NeurIPS 2022 Datasets and Benchmarks), the one TorchDrug's ``SubcellularLocalization`` loads: DeepLoc's test set,
    and its training set split into train/valid. The released files hold 8,420 / 2,811 / 2,773 proteins (the paper's
    table says 8,945 / 2,248 / 2,768). Labels follow PEER's order (``LOCATIONS``); DeepLoc's "Cytoplasm-Nucleus"
    proteins are Cytoplasm.

    Structures are AlphaFold DB v4 predictions from the foldcomp database afdb_swissprot_v4; proteins without one are
    left out: 8,302 / 2,776 / 2,747 remain (13,825 of 14,004, 98.7%; the build script lists what is missing).
    ``scripts/build_deeploc.py`` makes the raw files: one json of records (``id``, ``label``, ``location``, ...) per
    split, and ``deeploc_structures.h5`` mapping each accession to its foldcomp-compressed structure. Graphs carry CA
    positions (``pos``), residue types (``x``), ``y``, ``seq`` and ``id``.
    """

    root = "data/deeploc"
    LOCATIONS = [
        "Cell.membrane",
        "Cytoplasm",
        "Endoplasmic.reticulum",
        "Golgi.apparatus",
        "Lysosome/Vacuole",
        "Mitochondrion",
        "Nucleus",
        "Peroxisome",
        "Plastid",
        "Extracellular",
    ]

    @property
    def raw_file_names(self):
        """Files that have to be present in the raw directory."""
        return ["deeploc_train.json", "deeploc_valid.json", "deeploc_test.json", "deeploc_structures.h5"]

    def download(self):
        """Raw files are built by a script, not fetched from W&B like the other downstream datasets."""
        missing = [f for f in self.raw_file_names if not os.path.exists(os.path.join(self.raw_dir, f))]
        raise FileNotFoundError(
            f"{type(self).__name__}: missing raw files in {self.raw_dir}: {', '.join(missing)}\n"
            "Build them with `python scripts/build_deeploc.py` (downloads DeepLoc, the PEER split and the ~3 GB "
            "afdb_swissprot_v4 foldcomp database)."
        )

    def _prepare_data(self, df: pd.DataFrame) -> List[Data]:
        data_list = []
        with h5py.File(self.raw_paths[3], "r") as h5:
            for row in tqdm(df.itertuples(), total=len(df)):
                seq, ca = foldcomp_ca(h5[row.id][()].tobytes())
                graph = Data(
                    x=torch.tensor([aminoacids(aa, "code") for aa in seq]),
                    pos=torch.from_numpy(ca),
                    y=torch.tensor(row.label, dtype=torch.long),
                    seq=seq,
                    id=row.id,
                )
                data_list.append(graph)
        return data_list


class DTIDataset(DownstreamDataset):
    """LP-PDBBind, molecules as ECFP."""

    root = "data/dti"

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

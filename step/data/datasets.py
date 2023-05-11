import os
import shutil
from pathlib import Path
from typing import Callable

import foldcomp
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data, Dataset, InMemoryDataset, download_url, extract_tar
from tqdm import tqdm

from .downstream import apply_edits, compute_edits
from .parsers import ProtStructure


class PreTrainDataset(Dataset):
    """
    Dataset for pre-training.
    """

    alphafold_db_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest"

    def __init__(
        self,
        root: str,
        transform: Callable = None,
        pre_transform: Callable = None,
        threads: int = 4,
    ):

        self._len = None
        self.threads = threads
        super().__init__(root, transform, pre_transform)

    def process_single_graph(self, idx: int, raw_path: Path) -> None:
        """Convert a single pdb file into a graph and save as .pt file."""
        s = ProtStructure(raw_path)
        data = Data(**s.get_graph(), uniprot_id=raw_path.stem)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(data, Path(self.processed_dir) / f"data_{idx}.pt")

    def process(self):
        """Convert pdbs into graphs."""
        paths = Path(self.raw_dir).glob("*.pdb")
        Parallel(n_jobs=self.threads)(delayed(self.process_single_graph)(idx, i) for idx, i in tqdm(enumerate(paths)))

    def get(self, idx: int):
        """Load a single graph."""
        return torch.load(self.processed_dir + "/" + f"data_{idx}.pt")

    def _set_len(self):
        """Calculating length each time is slow (~1s for 200k structures), setting it to an attribute."""
        self._len = len([x for x in Path(self.raw_dir).glob("*.pdb")])

    def len(self):
        """Number of graphs in the dataset."""
        if self._len is None:
            self._set_len()
        return self._len

    @property
    def raw_file_names(self):
        """List of raw files. I don't add pdb files, but create uniprot_ids.txt file."""
        return ["uniprot_ids.txt"]

    @property
    def processed_file_names(self):
        """All generated filenames."""
        return [f"data_{i}.pt" for i in range(self.len())]

    def download(self):
        """Download the dataset from alphafold DB."""
        print("Extracting the tar archives")
        Parallel(n_jobs=self.threads)(
            delayed(self.extract_tar)(tar_archive) for tar_archive in tqdm(Path(self.raw_dir).glob("*.tar"))
        )
        with open(os.path.join(self.raw_dir, "uniprot_ids.txt"), "w"):
            pass

    def extract_tar(self, i: Path):
        extract_tar(i, self.raw_dir, mode="r")
        i.unlink()


class FluorescenceDataset(InMemoryDataset):
    """One of the downstream tasks."""

    url = "http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/fluorescence.tar.gz"
    struct_url = "https://alphafold.ebi.ac.uk/files/AF-P42212-F1-model_v4.pdb"

    def __init__(self, root, split="train", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        """Download the dataset."""
        raw_dir = Path(self.raw_dir)
        download_url(self.url, self.raw_dir)
        download_url(self.struct_url, self.raw_dir)
        extract_tar(raw_dir / "fluorescence.tar.gz", self.raw_dir, mode="r")
        for i in (raw_dir / "fluorescence").glob("*.json"):
            shutil.move(i, self.raw_dir)
        (raw_dir / "fluorescence").rmdir()

    @property
    def raw_file_names(self):
        """Files that have to be present in the raw directory."""
        return [
            "fluorescence_train.json",
            "fluorescence_valid.json",
            "fluorescence_test.json",
            "AF-P42212-F1-model_v4.pdb",
        ]

    @property
    def processed_file_names(self):
        """Files that have to be present in the processed directory."""
        return ["train.pt", "valid.pt", "test.pt"]

    def process(self):
        """Do the full run for the dataset."""
        for split in ["train", "valid", "test"]:
            data_list = []
            if split == "train":
                idx = 0
            elif split == "valid":
                idx = 1
            elif split == "test":
                idx = 2
            df = pd.read_json(self.raw_paths[idx])
            df["primary"] = "M" + df["primary"]
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
                    **row["graph"].to_dict(),
                )
                for _, row in df.iterrows()
            ]

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[idx])


class StabilityDataset(InMemoryDataset):
    def __init__(self, root, split="train", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

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

    @property
    def processed_file_names(self):
        """Files that have to be present in the processed directory."""
        return ["train.pt", "valid.pt", "test.pt"]

    def process(self):
        """Do the full run for the dataset."""
        for split in ["train", "valid", "test"]:
            print(split)
            data_list = []
            if split == "train":
                idx = 0
            elif split == "valid":
                idx = 1
            elif split == "test":
                idx = 2
            df = pd.read_json(self.raw_paths[idx]).drop_duplicates(subset=["id"])
            ids = df["id"].tolist()
            df.set_index("id", inplace=True)

            with foldcomp.open(self.raw_paths[3], ids=ids) as db:
                for (name, pdb) in tqdm(db):
                    struct = ProtStructure(pdb)
                    graph = Data(**struct.get_graph())
                    graph["y"] = df.loc[name, "stability_score"][0]
                    data_list.append(graph)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[idx])

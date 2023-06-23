import os
import shutil
from pathlib import Path
from typing import Callable

import foldcomp
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset, download_url, extract_tar
from tqdm import tqdm

from .downstream import apply_edits, compute_edits
from .parsers import ProtStructure


class FoldSeekDataset(Dataset):
    """
    Dataset for pre-training.
    """

    def __init__(self, root: str = "data/foldseek", transform: Callable = None, pre_transform: Callable = None):
        super().__init__(root, transform, pre_transform)

    def process(self):
        """Convert proteins from foldcomp database into graphs."""
        idx = 0
        with foldcomp.open(self.raw_paths[0]) as db:
            for name, pdb in tqdm(db):
                if " " in name:
                    continue
                pdb = ProtStructure(pdb)
                if len(pdb) > 1022:
                    continue
                data = Data(**pdb.get_graph())
                split = name.split("-")
                if len(split) >= 2:
                    data.uniprot_id = split[1]
                else:
                    data.uniprot_id = name
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                torch.save(data, os.path.join(self.processed_dir, f"data_{idx}.pt"))
                idx += 1

    def get(self, idx):
        """Get graph by index."""
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        return data

    def len(self):
        """Number of graphs in the dataset."""
        return 2240576

    @property
    def raw_file_names(self):
        """The foldseek db."""
        return ["afdb_rep_v4", "afdb_rep_v4.dbtype", "afdb_rep_v4.index", "afdb_rep_v4.lookup", "afdb_rep_v4.source"]

    @property
    def processed_file_names(self):
        """All generated filenames."""
        return [f"data_{i}.pt" for i in range(0, self.len(), 1000)]


class FoldSeekSmallDataset(FoldSeekDataset):
    @property
    def raw_file_names(self):
        return ["e_coli", "e_coli.dbtype", "e_coli.index", "e_coli.lookup", "e_coli.source"]


class DownstreamDataset(InMemoryDataset):
    """Abstract class for downstream datasets."""

    splits = {"train": 0, "val": 1, "test": 2}
    root = None

    def __init__(self, split="train", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.splits[split]])

    @property
    def processed_file_names(self):
        """Files that have to be present in the processed directory."""
        return ["train.pt", "valid.pt", "test.pt"]


class FluorescenceDataset(DownstreamDataset):
    """Predict fluorescence for GFP mutants."""

    url = "http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/fluorescence.tar.gz"
    struct_url = "https://alphafold.ebi.ac.uk/files/AF-P42212-F1-model_v4.pdb"
    root = "data/fluorescence"

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
            "structures_db",
            "structures_db.index",
            "structures_db.lookup",
            "structures_db.dbtype",
        ]

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
                for name, pdb in tqdm(db):
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


class ESMDataset(DownstreamDataset):
    """
    Dataset using ESM embeddings.
    """

    root = None

    def process(self):
        for split in ["train", "valid", "test"]:
            data_list = []
            if split == "train":
                idx = 0
            elif split == "valid":
                idx = 1
            elif split == "test":
                idx = 2
            p = Path(self.raw_paths[idx])
            print(self.raw_paths)
            for i in tqdm(p.glob("*.pt")):
                entry = torch.load(i)
                data = Data(
                    x=entry["mean_representations"][33].unsqueeze(-1),
                    y=torch.tensor(float(entry["label"].split("_")[-1]), dtype=torch.float),
                )
                data_list.append(data)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ["train", "valid", "test"]


class FluorescenceESMDataset(ESMDataset):
    root = "data/fluorescence_esm"


class StabilityESMDataset(ESMDataset):
    root = "data/stability_esm"

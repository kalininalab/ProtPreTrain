import os
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data, Dataset, InMemoryDataset, extract_tar
from tqdm import tqdm


from .parsers import ProtStructure, aminoacids
from typing import List, Tuple


class PreTrainDataset(Dataset):
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


class TAPEDataset(InMemoryDataset):
    url = None

    def __init__(self, root, split="train", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            "fluorescence_train.json",
            "fluorescence_valid.json",
            "fluorescence_test.json",
            "AF-P42212-F1-model_v4.pdb",
        ]

    @property
    def processed_file_names(self):
        return ["train.pt", "valid.pt", "test.pt"]

    def process(self):
        for split in ["train", "valid", "test"]:
            data_list = []
            if split == "train":
                idx = 0
            elif split == "valid":
                idx = 1
            elif split == "test":
                idx = 2
            flup = pd.read_json(self.raw_paths[idx])
            flup['primary'] = 'M' + flup['primary']
            struct = ProtStructure(self.raw_paths[3])
            pdb_sequence = struct.get_sequence()
            graph = Data(**struct.get_graph())
            for i in range(len(flup)):
                changes = find_changes(pdb_sequence, flup.loc[i, "primary"])
                data = apply_changes(graph, changes)
                data.Y = flup.loc[i, "log_fluorescence"][0]
                data_list.append(data)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[idx])

def find_changes(orig_seq: str, new_seq: str) -> List[Tuple[str, int, str]]:
    """Find changes between two sequences."""
    n1, n2 = len(orig_seq), len(new_seq)
    assert n1 == n2, f"Sequences must be of the same length, got {n1} and {n2}"
    changes = []
    for i, (a,b) in enumerate(zip(orig_seq, new_seq)):
        if a != b:
            changes.append((a, i, b))
    return changes

def apply_changes(graph: Data, changes: List[Tuple[str, int, str]]) -> Data:
    """Apply changes to a graph."""
    new_graph = graph.clone()
    for change in changes:
        new_graph.x[change[1]] = aminoacids(change[2], "code")
    return new_graph
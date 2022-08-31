import os
import shutil
from pathlib import Path
from typing import Callable

import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data, Dataset, download_url, extract_gz, extract_tar
from tqdm import tqdm

from .pdb_parser import PDBStructure


class AlphaFoldDataset(Dataset):
    alphafold_db_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest"

    def __init__(
        self,
        root: str,
        archive: str,
        transform: Callable = None,
        pre_transform: Callable = None,
        threads: int = 4,
    ):
        self._len = None
        self.threads = threads
        self.archive = archive
        super().__init__(root, transform, pre_transform)

    def process_single_graph(self, idx: int, raw_path: Path) -> None:
        """Convert a single pdb file into a graph and save as .pt file."""
        s = PDBStructure(raw_path)
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
        shutil.rmtree(self.raw_dir)
        download_url(os.path.join(self.alphafold_db_url, self.archive), self.raw_dir)
        extract_tar(os.path.join(self.raw_dir, self.archive), self.raw_dir, mode="r")
        filenames = []
        for i in Path(self.raw_dir).glob("*.pdb.gz"):
            extract_gz(i, self.raw_dir)
            i.unlink()  # remove gz files
            filenames.append(str(i))
        for i in Path(self.raw_dir).glob("*.cif.gz"):  # If cif files are downloaded, remove them
            i.unlink()
        os.unlink(os.path.join(self.raw_dir, self.archive))
        with open(os.path.join(self.raw_dir, "uniprot_ids.txt"), "w"):
            pass

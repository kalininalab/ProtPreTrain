import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch


class AAFrame(pd.DataFrame):
    def __call__(self, value: Union[str, int], target: str) -> Union[str, int]:
        """Return correct representation of the aminoacid"""
        assert target in self.columns, "Target columns must be in the dataframe"
        if isinstance(value, str):
            value = value.upper()
        if isinstance(value, int):
            source = "code"
        elif isinstance(value, str):
            if len(value) == 1:
                source = "one"
            elif len(value) == 3:
                source = "three"
            else:
                raise ValueError("Aminoacid must be either 1 or 3 letter code")
        else:
            raise ValueError("Aminoacid must be int or str")
        return self.set_index(source).loc[value, target]


aminoacids = AAFrame(
    [
        [0, "ALA", "A"],
        [1, "ARG", "R"],
        [2, "ASN", "N"],
        [3, "ASP", "D"],
        [4, "CYS", "C"],
        [5, "GLN", "Q"],
        [6, "GLU", "E"],
        [7, "GLY", "G"],
        [8, "HIS", "H"],
        [9, "ILE", "I"],
        [10, "LEU", "L"],
        [11, "LYS", "K"],
        [12, "MET", "M"],
        [13, "PHE", "F"],
        [14, "PRO", "P"],
        [15, "SER", "S"],
        [16, "THR", "T"],
        [17, "TRP", "W"],
        [18, "TYR", "Y"],
        [19, "VAL", "V"],
        [20, "MASK", "X"],
    ],
    columns=["code", "three", "one"],
)


class Residue:
    """Residue class"""

    def __init__(self, name: str, x: float, y: float, z: float):
        self.name = name.lower()
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_pdb_line(cls, line: str):
        return cls(
            name=line[17:20].strip(),
            x=float(line[30:38]),
            y=float(line[38:46]),
            z=float(line[46:54]),
        )

    @classmethod
    def from_cif_line(cls, line: str):
        return cls(
            name=line[20:23].strip(),
            x=float(line[34:42].strip()),
            y=float(line[42:50].strip()),
            z=float(line[50:58].strip()),
        )


class ProtStructure:
    """Structure class"""

    def __init__(self, filename: str) -> None:
        self.residues = []
        suffixes = Path(filename).suffixes
        if ".gz" in suffixes:
            f = gzip.open(filename, "rt")
        else:
            f = open(filename, "r")
        if ".pdb" in suffixes:
            self.parse_pdb_file(f)
        elif ".cif" in suffixes:
            self.parse_cif_file(f)
        else:
            raise ValueError(f"Unknown file extension {''.join(suffixes)}")

    def parse_pdb_file(self, f) -> None:
        """Parse PDB file"""
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res = Residue.from_pdb_line(line)
                self.residues.append(res)

    def parse_cif_file(self, f) -> None:
        """Parse CIF file"""
        for line in f:
            if line.startswith("ATOM") and line[14:18].strip() == "CA":
                res = Residue.from_cif_line(line)
                self.residues.append(res)

    def get_coords(self) -> torch.Tensor:
        """Get coordinates of all atoms"""
        coords = [[res.x, res.y, res.z] for res in self.residues]
        return torch.tensor(coords)

    def get_nodes(self) -> torch.Tensor:
        """Get features of all nodes of a graph"""
        return torch.tensor([aminoacids(res.name, "code") for res in self.residues])

    def get_edges(self, threshold: float) -> torch.Tensor:
        """Get edges of a graph using threshold as a cutoff"""
        coords = self.get_coords()
        dist = torch.cdist(coords, coords)
        edges = torch.where(dist < threshold)
        edges = torch.cat([arr.view(-1, 1) for arr in edges], axis=1)
        edges = edges[edges[:, 0] != edges[:, 1]]
        return edges.t()

    def get_graph(self) -> dict:
        """Get a point cloud representation of a protein."""
        nodes = []
        pos = []
        for res in self.residues:
            nodes.append(aminoacids(res.name, "code"))
            pos.append([res.x, res.y, res.z])
        return dict(x=torch.tensor(nodes), pos=torch.tensor(pos))
    
    def get_sequence(self) -> str:
        """Get sequence of a protein"""
        return "".join([aminoacids(res.name, "one") for res in self.residues])

    def __len__(self):
        return len(self.residues)

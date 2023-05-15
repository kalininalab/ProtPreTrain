import gzip
import os
from pathlib import Path
from typing import List, Union

import torch

ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "MASK",
}

THREE_TO_ONE = {v: k for k, v in ONE_TO_THREE.items()}

THREE_TO_CODE = {v: i for i, (k, v) in enumerate(ONE_TO_THREE.items())}

CODE_TO_THREE = {v: k for k, v in THREE_TO_CODE.items()}


def aminoacids(value: Union[str, int], target: str) -> Union[str, int]:
    """Converts between one-letter, three-letter, and code representations of amino acids."""
    if isinstance(value, str):
        value = value.upper()
    if target == "one":
        if len(value) == 1:
            return value
        elif len(value) == 3:
            return THREE_TO_ONE[value]
    elif target == "three":
        if isinstance(value, int):
            return CODE_TO_THREE[value]
        elif len(value) == 3:
            return value
        elif len(value) == 1:
            return ONE_TO_THREE[value]
    elif target == "code":
        if len(value) == 3:
            return THREE_TO_CODE[value]
        elif len(value) == 1:
            return THREE_TO_CODE[ONE_TO_THREE[value]]
    else:
        raise ValueError("Target must be one of ['one', 'three', 'code']")


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
    """Protein structure class."""

    def __init__(self, input_data: str | Path) -> None:
        self.residues = []
        if check_path_valid(input_data):  # valid path == file
            suffixes = Path(input_data).suffixes
            if ".gz" in suffixes:  # if the file is gzipped
                f = gzip.open(input_data, "rt")
            else:
                f = open(input_data, "r")
            content = f.read()
            f.close()
        else:  # invalid path == string
            content = input_data
        file_format = check_file_format(content)
        if file_format == "pdb":
            self.parse_pdb(content)
        elif file_format == "cif":
            self.parse_cif(content)
        else:
            raise ValueError("File format not recognized")

    def parse_pdb(self, content: List[str]) -> None:
        """Parse PDB file"""
        for line in content.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res = Residue.from_pdb_line(line)
                self.residues.append(res)

    def parse_cif(self, content: List[str]) -> None:
        """Parse CIF file"""
        for line in content.splitlines():
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


def check_path_valid(path: str):
    """Check if path is valid."""
    max_path_length = os.pathconf(path, "PC_PATH_MAX")
    if len(path) > max_path_length:
        return False
    p = Path(path)
    return p.is_file()


def check_file_format(content: str) -> str:
    """Check if the content is PDB or CIF format."""
    pdb_keywords = ["ATOM", "HETATM", "TER", "MODEL", "ENDMDL", "HEADER", "COMPND"]
    cif_keywords = ["_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol"]

    pdb_count = sum([1 for keyword in pdb_keywords if keyword in content])
    cif_count = sum([1 for keyword in cif_keywords if keyword in content])

    if pdb_count > 0 and cif_count == 0:
        return "pdb"
    elif cif_count > 0 and pdb_count == 0:
        return "cif"
    else:
        return None

import gzip
from pathlib import Path

import torch

node_encode = {
    "ala": 0,
    "arg": 1,
    "asn": 2,
    "asp": 3,
    "cys": 4,
    "gln": 5,
    "glu": 6,
    "gly": 7,
    "his": 8,
    "ile": 9,
    "leu": 10,
    "lys": 11,
    "met": 12,
    "phe": 13,
    "pro": 14,
    "ser": 15,
    "thr": 16,
    "trp": 17,
    "tyr": 18,
    "val": 19,
    "mask": 20,
}


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
        return torch.tensor([node_encode[res.name.lower()] for res in self.residues])

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
            nodes.append(node_encode[res.name.lower()])
            pos.append([res.x, res.y, res.z])
        return dict(x=torch.tensor(nodes), pos=torch.tensor(pos))

    def __len__(self):
        return len(self.residues)

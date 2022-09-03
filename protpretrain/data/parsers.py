import gzip
from pathlib import Path

import torch
from gemmi import cif

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

    def __init__(self, name: str, x: float, y: float, z: float, plddt: float = 100.0):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.plddt = plddt

    @classmethod
    def from_pdb_line(cls, line: str):
        cls(
            name=line[17:20].strip(),
            x=float(line[30:38]),
            y=float(line[38:46]),
            z=float(line[46:54]),
            plddt=float(line[60:66]),
        )

    @classmethod
    def from_cif_row(cls, row: cif.BlockRow):
        return cls(
            name=row["label_comp_id"],
            x=float(row["Cartn_x"]),
            y=float(row["Cartn_y"]),
            z=float(row["Cartn_z"]),
            plddt=float(row["plddt"]),
        )


class ProtStructure:
    """Structure class"""

    def __init__(self, filename: str) -> None:
        self.residues = []
        suffixes = Path(filename).suffixes
        if ".pdb" in suffixes:
            self.parse_pdb_file(filename)
        else:
            self.parse_cif_file(filename)
        self.parse_file(filename)

    def parse_pdb_file(self, filename: str) -> None:
        """Parse PDB file"""
        if ".gz" in filename:
            f = gzip.open(filename, "rt")
        else:
            f = open(filename, "r")
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res = Residue(line)
                self.residues.append(res)

    def parse_cif_file(self, filename: str) -> None:
        """Parse CIF file"""
        block = cif.read(filename).sole_block()
        for row in block.find_values("_atom_site", "ATOM"):
            if row["_atom_site.label_atom_id"] == "CA":
                res = Residue.from_cif_row(row)
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
        """Get a graph using threshold as a cutoff"""
        nodes = []
        pos = []
        plddt = []
        for res in self.residues.values():
            nodes.append(node_encode[res.name.lower()])
            pos.append([res.x, res.y, res.z])
            plddt.append(res.plddt)
        return dict(x=torch.tensor(nodes), pos=torch.tensor(pos), plddt=torch.tensor(plddt))


file = cif.read_file("../5tvn.cif")
greeted = set()
block = cif.read_file("../5tvn.cif").sole_block()
coords = block.find("_atom_site.", ["label_atom_id", "label_comp_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z"])
for i in coords:
    if i["label_atom_id"] == "CA":
        print(i["Cartn_x"], i["Cartn_y"], i["Cartn_z"], i["label_comp_id"])

import re
from typing import List, Tuple

import foldcomp
import Levenshtein
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from .parsers import aminoacids


def compute_edits(seq1: str, seq2: str) -> List[Tuple[str, int, str]]:
    """Compute edits to transform seq1 into seq2."""
    edit_operations = Levenshtein.editops(seq1, seq2)
    edits = []
    for op, pos1, pos2 in reversed(edit_operations):
        if op == "replace":
            edits.append(("replace", pos1, seq2[pos2]))
        elif op == "insert":
            edits.append(("insert", pos1, seq2[pos2]))
        elif op == "delete":
            edits.append(("delete", pos1, "X"))
    return edits


def delete_row(tensor: torch.Tensor, row_index: int) -> torch.Tensor:
    """Delete a row from a tensor."""
    return torch.cat([tensor[:row_index], tensor[row_index + 1 :]])


def apply_edits(protein: Data, edit_operations: List[Tuple[str, int, str]]) -> Data:
    """Apply edits to a protein. The edits are calculated with `compute_edits`."""
    dataset_len = protein.x.size(0)
    mutant = protein.clone()
    for op, idx, aa in edit_operations:
        new_x = aminoacids(aa, "code")
        if op == "replace":
            mutant.x[idx] = new_x
        elif op == "insert":
            new_pos = (mutant.pos[max(idx - 1, 0)] + mutant.pos[min(idx, dataset_len - 1)]) / 2
            mutant.x = torch.cat([mutant.x[:idx], mutant.x.new_tensor([new_x]), mutant.x[idx:]])
            mutant.pos = torch.cat([mutant.pos[:idx], new_pos.unsqueeze(0), mutant.pos[idx:]])
        elif op == "delete":
            mutant.x = delete_row(mutant.x, idx)
            mutant.pos = delete_row(mutant.pos, idx)
    return mutant


def extract_uniprot_id(title: str) -> str:
    """Extract the UniProt ID from a foldcomp name."""
    pattern1 = r"\(([A-Z0-9]{6,})\)"
    if " " in title:
        match1 = re.search(pattern1, title)
        return match1.group(1)
    elif title.startswith("AF-"):
        return title.split("-")[1]
    else:
        raise ValueError(f"Title '{title}' does not match any pattern.")


# Heavy atoms per residue, in the order foldcomp.get_data lists coordinates: N, CA, C, O, then the side chain
HEAVY_ATOMS = dict(
    A=5, R=11, N=8, D=8, C=6, Q=9, E=9, G=4, H=10, I=8, L=8, K=9, M=8, F=11, P=7, S=6, T=7, W=14, Y=12, V=7
)


def foldcomp_ca(fcz: bytes) -> Tuple[str, np.ndarray]:
    """Sequence and CA coordinates (float32, N x 3) of a foldcomp-compressed structure.

    Uses foldcomp.get_data rather than foldcomp.decompress + ProtStructure: decompress leaks the PDB text it returns
    (~300 KB per structure, GBs over a dataset). Same result as parsing the decompressed PDB's CA atoms.
    """
    data = foldcomp.get_data(fcz)
    seq = data["residues"]
    counts = np.array([HEAVY_ATOMS[aa] for aa in seq])
    coords = np.asarray(data["coordinates"], dtype=np.float32)
    if len(coords) != counts.sum() + 1:  # + the C-terminal OXT
        raise ValueError(f"{len(coords)} atoms for a {len(seq)}-residue chain, expected {counts.sum() + 1}")
    return seq, coords[np.cumsum(counts) - counts + 1]


def smiles_to_ecfp(smiles: str, radius: int = 2, nbits: int = 2048) -> torch.Tensor:
    """Convert a SMILES string to an ECFP fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
    ecfp_tensor = torch.tensor(list(ecfp), dtype=torch.float32)
    return ecfp_tensor

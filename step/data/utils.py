import os
import re
import shutil
from math import ceil
from pathlib import Path
from typing import Iterable, List, Tuple

import Levenshtein
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
    indices = torch.tensor([i for i in range(tensor.size(0)) if i != row_index])
    return torch.index_select(tensor, 0, indices)


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
            mutant.x = torch.cat([mutant.x[:idx], torch.tensor([new_x]), mutant.x[idx:]])
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


def save_file(data_list: list, filename: str):
    """While file is saving call in .tmp file, then rename to original filename."""
    temp_filename = Path(filename).with_suffix(".tmp")
    torch.save(data_list, temp_filename)
    os.rename(temp_filename, filename)


def split_indices_between_workers(indices: list, num_workers: int) -> list[list[int]]:
    """Split indices between workers."""
    k = ceil(len(indices) / num_workers)
    l = [indices[x : x + k] for x in range(0, len(indices), k)]
    return l


def smiles_to_ecfp(smiles: str, radius: int = 2, nBits: int = 2048) -> torch.Tensor:
    """Convert a SMILES string to an ECFP fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    ecfp_tensor = torch.tensor(list(ecfp))
    return ecfp_tensor

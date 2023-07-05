from typing import List, Tuple

import Levenshtein
import torch
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
    n = protein.x.size(0)
    mutant = protein.clone()
    for op, idx, aa in edit_operations:
        new_x = aminoacids(aa, "code")
        if op == "replace":
            mutant.x[idx] = new_x
        elif op == "insert":
            new_pos = (mutant.pos[max(idx - 1, 0)] + mutant.pos[min(idx, n - 1)]) / 2
            mutant.x = torch.cat([mutant.x[:idx], torch.tensor([new_x]), mutant.x[idx:]])
            mutant.pos = torch.cat([mutant.pos[:idx], new_pos.unsqueeze(0), mutant.pos[idx:]])
        elif op == "delete":
            mutant.x = delete_row(mutant.x, idx)
            mutant.pos = delete_row(mutant.pos, idx)
    return mutant
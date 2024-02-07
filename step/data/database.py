import sqlite3
from torch_geometric.data import Data
import torch
import numpy as np


class Database:

    def __init__(self, filename: str):
        self.filename = filename
        self.name = "FoldCompDataset"
        self.conn = sqlite3.connect(filename)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {self.name} (
            id INTEGER PRIMARY KEY,
            x BLOB NOT NULL,
            edge_index BLOB NOT NULL,
            pos BLOB NOT NULL,
            pe BLOB NOT NULL,
            uniprot_id TEXT
        )
        """
        )
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx ON {self.name} (id)")
        self.conn.commit()

    def insert(self, index: int, data: dict) -> None:
        query = f"INSERT INTO {self.name} (id, x, edge_index, pos, pe, uniprot_id) VALUES (?, ?, ?, ?, ?, ?)"
        self.cursor.execute(
            query,
            (
                index,
                data["x"].numpy().tobytes(),
                data["edge_index"].numpy().tobytes(),
                data["pos"].numpy().tobytes(),
                data["pe"].numpy().tobytes(),
                data["uniprot_id"],
            ),
        )
        self.conn.commit()

    def multi_insert(self, idx_list: list[int], data: list[dict]) -> None:
        query = f"INSERT INTO {self.name} (id, x, edge_index, pos, pe, uniprot_id) VALUES (?, ?, ?, ?, ?, ?)"
        self.cursor.executemany(
            query,
            [
                (
                    idx,
                    data["x"].numpy().tobytes(),
                    data["edge_index"].numpy().tobytes(),
                    data["pos"].numpy().tobytes(),
                    data["pe"].numpy().tobytes(),
                    data["uniprot_id"],
                )
                for idx, data in zip(idx_list, data)
            ],
        )
        self.conn.commit()

    def get(self, idx: int) -> Data:
        self.cursor.execute(f"SELECT * FROM FoldCompDataset WHERE id={idx}")
        row = self.cursor.fetchone()
        data = Data(
            x=torch.frombuffer(row[1], dtype=torch.long).clone(),
            edge_index=torch.frombuffer(row[2], dtype=torch.long).clone(),
            pos=torch.frombuffer(row[3], dtype=torch.float32).clone(),
            pe=torch.frombuffer(row[4], dtype=torch.float32).clone(),
            uniprot_id=row[5],
        )
        return data

    def __len__(self):
        self.cursor.execute(f"SELECT COUNT(id) FROM {self.name}")
        return self.cursor.fetchone()[0]

    def close(self):
        self.conn.close()

    def get_all_indices(self):
        self.cursor.execute(f"SELECT id FROM {self.name}")
        return [row[0] for row in self.cursor.fetchall()]

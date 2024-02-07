import sqlite3
from torch_geometric.data import Data
import torch
import numpy as np


class Database:

    def __init__(self, filename: str, schema: dict):
        self.filename = filename
        self.name = "FoldCompDataset"
        self.conn = sqlite3.connect(filename)
        self.cursor = self.conn.cursor()
        self.schema = schema
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.name} ({self._schema_to_sql()})")
        self.conn.commit()

    def _dummies(self):
        return ",".join(["?"] * len(self.schema))

    def _schema_to_sql(self) -> str:
        res = ["id INTEGER PRIMARY KEY"]
        for k, v in self.schema.items():
            if isinstance(v, int):
                dtype = "INTEGER"
            elif isinstance(v, str):
                dtype = "TEXT"
            else:
                dtype = "BLOB NOT NULL"
            res.append(f"{k} {dtype}")
        return ", ".join(res)

    def _serialize(self, data: dict) -> list:
        row = []
        for k in self.schema.keys():
            if isinstance(data[k], torch.Tensor):
                row.append(data[k].numpy().tobytes())
            else:
                row.append(data[k])
        return row

    def _deserialize(self, row: tuple) -> Data:
        res = {}
        for idx, (k, v) in enumerate(self.schema.items()):
            if isinstance(v, torch.Tensor):
                res[k] = torch.frombuffer(row[idx + 1], dtype=v)
            else:
                res[k] = row[idx + 1]
        return Data.from_dict(res)

    def multi_insert(self, idx_list: list[int], data: list[dict]) -> None:
        query = f"INSERT INTO {self.name} (id, {','.join(self.schema.keys())}) VALUES (?, {self._dummies()})"
        data_list = []
        for idx, data in zip(idx_list, data):
            row = self._serialize(data)
            data_list.append((idx, *row))
            print(idx)
        self.cursor.executemany(query, data_list)
        self.conn.commit()

    def multi_get(self, idx) -> list:
        if isinstance(idx, int):
            idx_as_list = [idx]
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            idx_as_list = list(range(start, stop, step))
        elif isinstance(idx, list):
            idx_as_list = idx
        query = f"SELECT * FROM FoldCompDataset WHERE id IN ({','.join('?' * len(idx_as_list))})"
        self.cursor.execute(query, idx_as_list)
        rows = self.cursor.fetchall()
        return [self._deserialize(row) for row in rows]

    def __getitem__(self, idx):
        return self.multi_get(idx)

    def __len__(self):
        self.cursor.execute("SELECT max(id) FROM FoldCompDataset")
        return self.cursor.fetchone()[0]

    def close(self):
        self.conn.close()

import sqlite3

import torch
from torch_geometric.data import Data, SQLiteDatabase


class Database(SQLiteDatabase):
    """Database for FoldCompDataset. It uses sqlite3 to store the data."""

    def __init__(self, filename: str, schema: dict):
        self.filename = filename
        self.name = "FoldCompDataset"
        self.conn = sqlite3.connect(filename)
        self.cursor = self.conn.cursor()
        self.schema = schema
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.name} ({self._schema_to_sql()})")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx ON {self.name} (id)")
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

    def multi_insert(self, idx_list: list[int], data: list[dict]) -> None:
        """Insert multiple rows into the database."""
        query = f"INSERT INTO {self.name} (id, {','.join(self.schema.keys())}) VALUES (?, {self._dummies()})"
        data_list = []
        for idx, data in zip(idx_list, data):
            row = self._serialize(data)
            print((idx, *row))
            raise ValueError
            data_list.append((idx, *row))
        self.cursor.executemany(query, data_list)
        self.conn.commit()

    def multi_get(self, idx) -> list:
        """Get multiple rows from the database."""
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
        self.cursor.execute(f"SELECT COUNT(id) FROM {self.name}")
        return self.cursor.fetchone()[0]

    def get_all_indices(self) -> list[int]:
        """Get all indices from the database."""
        self.cursor.execute(f"SELECT id FROM {self.name}")
        return [row[0] for row in self.cursor.fetchall()]

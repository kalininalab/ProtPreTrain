"""Build the DeepLoc 10-class subcellular localization dataset (PEER split) with AlphaFold structures.

    python scripts/build_deeploc.py

Writes the raw files step.data.DeepLocDataset reads to data/deeploc/raw/:

- ``deeploc_{train,valid,test}.json``: one record per protein, ``id`` (UniProt accession), ``label`` (0-9),
  ``location`` (class name), ``membrane`` (DeepLoc's M/S/U membrane-bound annotation) and ``sequence`` (DeepLoc's).
- ``deeploc_structures.h5``: ``id`` -> foldcomp-compressed AlphaFold structure (uint8 array, read back with
  ``foldcomp.decompress(h5[id][()].tobytes())``).

The split is PEER's (Xu et al. 2022), the one TorchDrug's ``SubcellularLocalization`` loads: DeepLoc 1.0's test set,
and its training set split into train/valid. PEER ships only sequences, so accessions are recovered by matching
against DeepLoc 1.0's FASTA (PEER cut sequences longer than 1000 residues to their first and last 500). Structures
come from the foldcomp database afdb_swissprot_v4 (downloaded on first use, ~3 GB, to data/afdb_swissprot_v4/raw);
proteins it has no model for are dropped. On the 2026-10 build that is 179 of 14,004: 75 isoforms (``P12345-2``;
AlphaFold DB models canonical sequences only), 9 accessions UniProt has since demerged or deleted, and 95 current
entries without a model (61 of the 104 non-isoforms are longer than 2,700 residues, AlphaFold DB's single-model
limit). None of the 179 is a merged accession whose new entry could stand in. 141 of the kept models' sequences differ
from DeepLoc's (UniProt sequence updates since 2016); graphs and ``seq`` always come from the structure.

Records are shuffled within each split (fixed seed): PEER sorts them by label, so an unshuffled val/test loader would
see single-class batches. Idempotent: downloads already present are reused, and the outputs are rewritten whole (and
identically) on every run.
"""

import argparse
import collections
import json
import os
import pickle
import random
import struct
import tarfile
import urllib.request
from pathlib import Path

import foldcomp
import h5py
import numpy as np

from step.data.datasets import DeepLocDataset
from step.data.utils import foldcomp_ca

DEEPLOC_URL = "https://services.healthtech.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta"
PEER_URL = (
    "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization.tar.gz"
)
USER_AGENT = "STEP-build_deeploc/1.0 (https://github.com/kalininalab/ProtPreTrain)"

# PEER's label order. DeepLoc's "Cytoplasm-Nucleus" proteins are labelled Cytoplasm (checked while matching)
LOCATIONS = DeepLocDataset.LOCATIONS
SPLITS = ["train", "valid", "test"]


def download(url: str, path: Path) -> Path:
    """Download url to path unless it already exists; written to a .part file first, so a crash leaves no stub."""
    if path.exists():
        return path
    print(f"Downloading {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


def read_lmdb(buf: bytes):
    """Yield (key, value) of the main database of a 64-bit LMDB ``data.mdb``, walking the B-tree from the newest meta.

    A minimal reader, so the build needs no ``lmdb`` package: it covers what PEER's files use (no sub-databases, no
    duplicate keys). Verified against the lmdb library on all three PEER splits.
    """
    metas = []
    for pgno in (0, 1):  # two meta pages, the one with the higher txnid is current
        off = pgno * 4096 + 16
        magic, _version, _addr, _mapsize, psize = struct.unpack_from("<IIQQI", buf, off)
        assert magic == 0xBEEFC0DE, f"not an LMDB file (magic {magic:#x})"
        root, _last_pg, txnid = struct.unpack_from("<QQQ", buf, off + 24 + 48 + 40)
        metas.append((txnid, psize, root))
    _, psize, root = max(metas)

    def walk(pgno: int):
        base = pgno * psize
        flags, lower = struct.unpack_from("<HH", buf, base + 10)
        for i in range((lower - 16) // 2):
            node = base + struct.unpack_from("<H", buf, base + 16 + 2 * i)[0]
            lo, hi, node_flags, ksize = struct.unpack_from("<HHHH", buf, node)
            if flags & 0x01:  # branch page: the node holds a child page number
                yield from walk(lo | hi << 16 | node_flags << 32)
                continue
            assert flags & 0x02, f"unexpected LMDB page flags {flags:#x}"
            key = buf[node + 8 : node + 8 + ksize]
            data = node + 8 + ksize
            if node_flags & 0x01:  # big value, stored on its own overflow pages
                data = struct.unpack_from("<Q", buf, data)[0] * psize + 16
            yield key, buf[data : data + (lo | hi << 16)]

    if root != 2**64 - 1:  # empty database
        yield from walk(root)


def read_peer(tar_path: Path) -> dict:
    """PEER split -> list of (sequence, label), in record order."""
    splits = {}
    with tarfile.open(tar_path) as tar:
        for split in SPLITS:
            buf = tar.extractfile(f"subcellular_localization/subcellular_localization_{split}.lmdb/data.mdb").read()
            records = dict(read_lmdb(buf))
            n = pickle.loads(records.pop(b"num_examples"))
            # PEER's own pickles of plain dicts (str + int), from the URL above
            items = [pickle.loads(records[str(i).encode()]) for i in range(n)]
            assert len(records) == n, f"{split}: {len(records)} records, num_examples {n}"
            splits[split] = [(item["primary"], int(item["localization"])) for item in items]
    return splits


def read_deeploc(fasta: Path) -> list:
    """DeepLoc 1.0 FASTA -> list of dicts (accession, location, membrane, test flag, sequence), in file order."""
    records = []
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            acc, label, *test = line[1:].split()
            location, membrane = label.rsplit("-", 1)
            records.append(dict(id=acc, location=location, membrane=membrane, test=test == ["test"], sequence=""))
        elif line.strip():
            records[-1]["sequence"] += line.strip()
    return records


def peer_key(seq: str) -> str:
    """PEER's truncation of sequences longer than 1000 residues: the first and last 500."""
    return seq[:500] + seq[-500:] if len(seq) > 1000 else seq


def match_splits(peer: dict, deeploc: list) -> dict:
    """Assign every PEER record a DeepLoc entry, one to one; identical sequences are assigned in file order."""
    pool = collections.defaultdict(collections.deque)
    for rec in deeploc:
        pool[peer_key(rec["sequence"])].append(rec)
    out = {}
    for split, items in peer.items():
        out[split] = []
        for seq, label in items:
            candidates = pool[seq]
            rec = next(r for r in candidates if r["test"] == (split == "test"))
            candidates.remove(rec)
            location = "Cytoplasm" if rec["location"] == "Cytoplasm-Nucleus" else rec["location"]
            assert location == LOCATIONS[label], f"{rec['id']}: PEER label {label}, DeepLoc {rec['location']}"
            out[split].append(
                dict(
                    id=rec["id"],
                    label=label,
                    location=LOCATIONS[label],
                    membrane=rec["membrane"],
                    sequence=rec["sequence"],
                )
            )
    left = sum(len(v) for v in pool.values())
    assert left == 0, f"{left} DeepLoc entries not in any PEER split"
    return out


def ensure_foldcomp_db(db: Path) -> None:
    """Download a foldcomp database with foldcomp.setup (resumes partial downloads) unless it is complete."""
    if all(Path(f"{db}{ext}").exists() for ext in ["", ".index", ".dbtype", ".lookup"]):
        return
    print(f"Downloading foldcomp database {db.name} into {db.parent} (~3 GB for afdb_swissprot_v4)")
    db.parent.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(db.parent)
    try:
        foldcomp.setup(db.name)
    finally:
        os.chdir(cwd)


def fetch_structures(db: Path, accessions: list) -> dict:
    """Accession -> compressed AlphaFold structure, for those in the foldcomp database (lookup AF-<acc>-F1-model_v4).

    Reads the MMseqs-style .lookup (key -> name) and .index (key -> offset, length) directly rather than through
    foldcomp.open: its reader drops each entry's last byte (the last residue's pLDDT; it takes it for an MMseqs NUL
    terminator, which foldcomp databases do not have), and naming entries by their PDB title needs foldcomp.decompress,
    which leaks its output (~300 KB per structure).
    """
    wanted = {f"AF-{acc}-F1-model_v4": acc for acc in accessions}
    keys = {}
    with open(f"{db}.lookup") as f:
        for line in f:
            key, name, *_ = line.rstrip("\n").split("\t")
            name = name.removesuffix(".pdb")  # some foldcomp databases keep the file extension in their lookup names
            if name in wanted:
                keys[int(key)] = wanted[name]
    found = {}
    with open(f"{db}.index") as index, open(db, "rb") as data:
        for line in index:
            key, offset, length = map(int, line.split())
            if key in keys:
                data.seek(offset)
                found[keys[key]] = data.read(length)
    return found


def main():
    """Download, match, fetch structures, write raw files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/deeploc", help="Dataset root; raw files go to <root>/raw")
    parser.add_argument(
        "--db", default="data/afdb_swissprot_v4/raw/afdb_swissprot_v4", help="foldcomp database, downloaded if absent"
    )
    args = parser.parse_args()

    root = Path(args.root)
    downloads = root / "download"
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    deeploc = read_deeploc(download(DEEPLOC_URL, downloads / "deeploc_data.fasta"))
    peer = read_peer(download(PEER_URL, downloads / "subcellular_localization.tar.gz"))
    splits = match_splits(peer, deeploc)
    print(
        f"DeepLoc 1.0: {len(deeploc)} proteins; PEER split: " + ", ".join(f"{s} {len(v)}" for s, v in splits.items())
    )

    db = Path(args.db)
    ensure_foldcomp_db(db)
    accessions = sorted({r["id"] for recs in splits.values() for r in recs})
    structures = fetch_structures(db, accessions)
    print(f"{len(structures)}/{len(accessions)} accessions found in {db.name}")

    missing = [acc for acc in accessions if acc not in structures]
    isoforms = sum("-" in acc for acc in missing)
    print(f"Dropped {len(missing)} without a structure ({isoforms} isoforms; AlphaFold DB models canonical sequences)")

    seq_match = 0
    out_h5 = raw / "deeploc_structures.h5"
    tmp_h5 = out_h5.with_name(out_h5.name + ".part")
    with h5py.File(tmp_h5, "w") as h5:
        for acc, fcz in sorted(structures.items()):
            h5.create_dataset(acc, data=np.frombuffer(fcz, dtype=np.uint8))
    tmp_h5.replace(out_h5)

    print(f"{'split':<6} {'DeepLoc':>8} {'with structure':>15}")
    for split, recs in splits.items():
        kept = [r for r in recs if r["id"] in structures]
        # PEER's records are sorted by label: unshuffled val/test batches would hold a single class each
        random.Random(0).shuffle(kept)
        for r in kept:  # also checks DeepLocDataset will be able to read every structure
            seq_match += foldcomp_ca(structures[r["id"]])[0] == r["sequence"]
        print(f"{split:<6} {len(recs):>8} {len(kept):>15}")
        counts = collections.Counter(r["location"] for r in kept)
        print("       " + ", ".join(f"{loc} {counts[loc]}" for loc in LOCATIONS))
        path = raw / f"deeploc_{split}.json"
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(json.dumps(kept))
        tmp.replace(path)
    total = sum(1 for recs in splits.values() for r in recs if r["id"] in structures)
    print(f"AlphaFold sequence identical to DeepLoc's for {seq_match}/{total} proteins")
    for path in sorted(raw.iterdir()):
        print(f"{path}: {path.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()

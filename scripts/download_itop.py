"""Download only ITOP depth maps and labels, then compact label metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from tqdm import tqdm

from data.itop_dataset import compact_itop_labels


ZENODO_RECORD_API = "https://zenodo.org/api/records/3932973"


class _ProgressReader:
    def __init__(self, response, total: int, description: str):
        self.response = response
        self.progress = tqdm(total=total, unit="B", unit_scale=True, desc=description)

    def read(self, size: int = -1):
        chunk = self.response.read(size)
        self.progress.update(len(chunk))
        return chunk

    def close(self):
        self.progress.close()
        self.response.close()


def _record_files() -> dict[str, dict]:
    with urllib.request.urlopen(ZENODO_RECORD_API) as response:
        record = json.load(response)
    return {entry["key"]: entry for entry in record["files"]}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(entry: dict, destination: Path) -> None:
    expected_size = int(entry["size"])
    expected_md5 = entry["checksum"].split(":", 1)[-1]
    if destination.exists() and destination.stat().st_size == expected_size:
        if _md5(destination) == expected_md5:
            print(f"verified existing {destination.name}")
            return
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(entry["links"]["self"]) as response:
        reader = _ProgressReader(response, expected_size, destination.name)
        try:
            with temporary.open("wb") as target:
                shutil.copyfileobj(reader, target, length=8 * 1024 * 1024)
        finally:
            reader.close()
    if temporary.stat().st_size != expected_size or _md5(temporary) != expected_md5:
        raise RuntimeError(f"checksum verification failed for {destination.name}")
    temporary.replace(destination)


def _decompress(source: Path, destination: Path) -> None:
    if destination.exists():
        print(f"keeping existing {destination.name}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    with gzip.open(source, "rb") as compressed, temporary.open("wb") as target:
        with tqdm(
            total=source.stat().st_size,
            unit="B",
            unit_scale=True,
            desc=f"decompress {source.name}",
        ) as progress:
            while True:
                chunk = compressed.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                progress.update(min(len(chunk), source.stat().st_size - progress.n))
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/itop")
    parser.add_argument("--view", choices=["side", "top", "all"], default="side")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--keep_full_labels", action="store_true")
    parser.add_argument("--delete_gzip", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    views = ("side", "top") if args.view == "all" else (args.view,)
    splits = ("train", "test") if args.split == "all" else (args.split,)
    record_files = _record_files()

    for view in views:
        for split in splits:
            for kind in ("depth_map", "labels"):
                name = f"ITOP_{view}_{split}_{kind}.h5.gz"
                compressed = data_dir / name
                _download(record_files[name], compressed)
                if args.download_only:
                    continue
                decompressed = compressed.with_suffix("")
                _decompress(compressed, decompressed)
                if kind == "labels":
                    compact = data_dir / f"ITOP_{view}_{split}_labels_compact.npz"
                    compact_itop_labels(decompressed, compact)
                    print(f"wrote {compact.name}")
                    if not args.keep_full_labels:
                        decompressed.unlink()
                        print(f"removed full label file {decompressed.name}")
                if args.delete_gzip:
                    compressed.unlink()
                    print(f"removed {compressed.name}")


if __name__ == "__main__":
    main()

"""Create an audited ModelNet40 cache without corrupted-scale point clouds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np

from data.paths import dataset_dir


RAW_CACHE_NAME = "modelnet40_inertia_dataset.pkl"
CLEAN_CACHE_NAME = "modelnet40_inertia_dataset_clean.pkl"
MAD_NORMAL_SCALE = 1.4826


def centered_max_radius(points: np.ndarray) -> np.ndarray:
    """Return a translation-, rotation-, and permutation-invariant scale."""
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [N, P, 3], got {points.shape}")
    centered = points - points.mean(axis=1, keepdims=True)
    return np.linalg.norm(centered, axis=-1).max(axis=1)


def fit_log_radius_threshold(
    train_points: np.ndarray,
    robust_z: float = 8.0,
) -> tuple[float, dict[str, float]]:
    """Fit an upper radius threshold using robust training-input statistics."""
    if robust_z <= 0.0:
        raise ValueError("robust_z must be positive")
    radii = centered_max_radius(train_points)
    log_radii = np.log(np.maximum(radii, np.finfo(np.float32).tiny))
    median = float(np.median(log_radii))
    mad = float(np.median(np.abs(log_radii - median)))
    robust_scale = MAD_NORMAL_SCALE * mad
    if robust_scale <= 0.0:
        raise ValueError("cannot fit a robust threshold because radius MAD is zero")
    threshold = float(np.exp(median + robust_z * robust_scale))
    return threshold, {
        "log_radius_median": median,
        "log_radius_mad": mad,
        "log_radius_robust_scale": robust_scale,
        "robust_z": float(robust_z),
        "radius_threshold": threshold,
    }


def training_target_statistics(targets: np.ndarray) -> dict[str, np.ndarray]:
    """Compute every cache statistic from the cleaned training split only."""
    targets = np.asarray(targets)
    dtype = targets.dtype

    def cast(value):
        return np.asarray(value, dtype=dtype)

    return {
        "mean": cast(targets.mean(axis=0)),
        "std": cast(targets.std(axis=0)),
        "min": cast(targets.min(axis=0)),
        "max": cast(targets.max(axis=0)),
        "median": cast(np.median(targets, axis=0)),
        "q25": cast(np.percentile(targets, 25, axis=0)),
        "q75": cast(np.percentile(targets, 75, axis=0)),
        "global_mean": cast(targets.mean()),
        "global_std": cast(targets.std()),
    }


def clean_cache_payload(
    payload: dict,
    *,
    robust_z: float = 8.0,
) -> tuple[dict, dict]:
    """Filter corrupted-scale samples and return cache plus audit metadata."""
    for split in ("train", "test"):
        if split not in payload:
            raise KeyError(f"cache is missing split {split!r}")
        for field in ("points", "inertia", "labels"):
            if field not in payload[split]:
                raise KeyError(f"cache split {split!r} is missing {field!r}")

    threshold, rule = fit_log_radius_threshold(
        payload["train"]["points"], robust_z=robust_z
    )
    cleaned: dict = {}
    split_audit: dict[str, dict] = {}
    for split in ("train", "test"):
        split_payload = payload[split]
        radii = centered_max_radius(split_payload["points"])
        keep = radii <= threshold
        sample_count = len(radii)
        cleaned_split = {}
        for name, value in split_payload.items():
            array = np.asarray(value)
            if array.ndim == 0 or len(array) != sample_count:
                raise ValueError(
                    f"{split}.{name} does not share the split length {sample_count}"
                )
            cleaned_split[name] = array[keep]
        cleaned[split] = cleaned_split

        removed_indices = np.flatnonzero(~keep)
        split_audit[split] = {
            "before": int(sample_count),
            "after": int(keep.sum()),
            "removed": [
                {
                    "index": int(index),
                    "label": int(split_payload["labels"][index]),
                    "centered_max_radius": float(radii[index]),
                    "max_abs_inertia": float(
                        np.abs(split_payload["inertia"][index]).max()
                    ),
                }
                for index in removed_indices
            ],
        }

    cleaned["stats"] = training_target_statistics(cleaned["train"]["inertia"])
    audit = {
        "version": 1,
        "selection_uses_targets": False,
        "rule": {
            "statistic": "max ||point - point_cloud_mean||_2",
            "fit_split": "train",
            "fit_space": "log radius",
            "threshold": "median + robust_z * 1.4826 * MAD",
            **rule,
        },
        "splits": split_audit,
        "statistics_fit_split": "cleaned train",
    }
    cleaned["cleaning"] = audit
    return cleaned, audit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_clean_cache(
    source: Path,
    destination: Path,
    *,
    robust_z: float,
) -> tuple[Path, Path]:
    source = source.resolve()
    destination = destination.resolve()
    audit_path = destination.with_suffix(".audit.json")
    if source == destination:
        raise ValueError("source and destination must differ")
    if destination.exists() or audit_path.exists():
        raise FileExistsError(f"refusing to overwrite {destination} or {audit_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = joblib.load(source)
    cleaned, audit = clean_cache_payload(payload, robust_z=robust_z)
    audit["source"] = str(source)
    audit["source_sha256"] = sha256(source)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(cleaned, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    audit["destination"] = str(destination)
    audit["destination_sha256"] = sha256(destination)
    audit["cleaned_training_statistics"] = {
        name: np.asarray(value).tolist() for name, value in cleaned["stats"].items()
    }
    with audit_path.open("x", encoding="utf-8") as target:
        json.dump(audit, target, indent=2)
    return destination, audit_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--robust_z", type=float, default=8.0)
    args = parser.parse_args()

    cache_dir = dataset_dir(None, "modelnet40") / "cache"
    source = Path(args.input_path) if args.input_path else cache_dir / RAW_CACHE_NAME
    destination = (
        Path(args.output_path) if args.output_path else cache_dir / CLEAN_CACHE_NAME
    )
    output, audit = write_clean_cache(source, destination, robust_z=args.robust_z)
    print(output)
    print(audit)


if __name__ == "__main__":
    main()

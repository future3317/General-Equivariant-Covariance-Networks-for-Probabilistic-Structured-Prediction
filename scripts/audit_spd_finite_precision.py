"""Audit mathematical versus finite-precision SPD behavior.

The typed compiler proves cone membership over real arithmetic.  This audit
checks the separate runtime question raised by extreme softplus logits: a
zero lower floor can underflow in reduced precision and turn a mathematical
SPD construction into a numerically singular matrix.  It deliberately does
not change any compiler default or add a hidden jitter.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from compatibility.e3nn import o3
from spd_maps import (
    IrrepBlockDiagonalMap,
    IsotropicMap,
    IsotypicBlockMap,
    LowRankPlusIsotropicMap,
)


def _case_factories() -> tuple[tuple[str, str, Callable[[torch.dtype], Any]], ...]:
    """Return representative zero-floor and positive-floor constructions."""
    irreps = o3.Irreps("2x0e+1x1o")
    return (
        (
            "isotropic",
            "zero_floor",
            lambda _dtype: IsotropicMap(dim=6, min_sigma2=0.0),
        ),
        (
            "isotropic",
            "positive_floor",
            lambda _dtype: IsotropicMap(dim=6, min_sigma2=1e-4),
        ),
        (
            "low_rank",
            "zero_floor",
            lambda _dtype: LowRankPlusIsotropicMap(
                dim=6, rank=2, min_sigma2=0.0
            ),
        ),
        (
            "low_rank",
            "positive_floor",
            lambda _dtype: LowRankPlusIsotropicMap(
                dim=6, rank=2, min_sigma2=1e-4
            ),
        ),
        (
            "isotypic_block",
            "zero_floor",
            lambda _dtype: IsotypicBlockMap(irreps, min_diagonal=0.0),
        ),
        (
            "isotypic_block",
            "positive_floor",
            lambda _dtype: IsotypicBlockMap(irreps, min_diagonal=1e-4),
        ),
        (
            "irrep_block_diagonal",
            "zero_floor",
            lambda _dtype: IrrepBlockDiagonalMap(irreps, min_sigma2=0.0),
        ),
        (
            "irrep_block_diagonal",
            "positive_floor",
            lambda _dtype: IrrepBlockDiagonalMap(irreps, min_sigma2=1e-4),
        ),
    )


def _parameters(name: str, map_object: Any, logit: float, dtype: torch.dtype) -> torch.Tensor:
    """Construct the smallest input that exercises a map's positive scalar."""
    if name == "isotropic":
        return torch.full((1, 1), logit, dtype=dtype)
    if name == "low_rank":
        params = torch.zeros((1, 6 * 2 + 1), dtype=dtype)
        params[..., -1] = logit
        return params
    if name == "isotypic_block":
        return torch.full((1, map_object.num_parameters), logit, dtype=dtype)
    if name == "irrep_block_diagonal":
        return torch.full((1, map_object.num_blocks), logit, dtype=dtype)
    raise ValueError(f"unknown SPD map {name!r}")


def _run_case(
    name: str,
    policy: str,
    factory: Callable[[torch.dtype], Any],
    dtype: torch.dtype,
    logit: float,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "map": name,
        "minimum_eigenvalue_policy": policy,
        "dtype": str(dtype).removeprefix("torch."),
        "logit": logit,
        "mathematical_cone_status": "strict_spd_for_finite_real_parameters",
    }
    try:
        map_object = factory(dtype)
        params = _parameters(name, map_object, logit, dtype)
        matrix = map_object(params)
        finite = bool(torch.isfinite(matrix).all())
        eigenvalues = torch.linalg.eigvalsh(matrix.float().double())
        minimum = float(eigenvalues.min().item())
        record.update(
            {
                "finite": finite,
                "minimum_eigenvalue": minimum,
                "finite_precision_cone_status": (
                    "strict_spd" if finite and minimum > 0.0 else "not_strict_spd"
                ),
            }
        )
    except (RuntimeError, ValueError, TypeError) as error:
        record.update(
            {
                "finite": False,
                "minimum_eigenvalue": None,
                "finite_precision_cone_status": "runtime_error",
                "error": f"{type(error).__name__}: {error}",
            }
        )
    return record


def run_audit(
    *,
    logits: tuple[float, ...] = (-100.0, -1_000.0, -10_000.0),
    dtypes: tuple[torch.dtype, ...] = (torch.float64, torch.float32, torch.bfloat16),
) -> dict[str, Any]:
    """Run the deterministic finite-precision cone audit on CPU."""
    records = [
        _run_case(name, policy, factory, dtype, logit)
        for name, policy, factory in _case_factories()
        for dtype in dtypes
        for logit in logits
    ]
    return {
        "schema_version": 1,
        "device": "cpu",
        "logits": list(logits),
        "dtypes": [str(dtype).removeprefix("torch.") for dtype in dtypes],
        "records": records,
        "summary": {
            "total": len(records),
            "strict_spd": sum(
                record["finite_precision_cone_status"] == "strict_spd"
                for record in records
            ),
            "not_strict_spd": sum(
                record["finite_precision_cone_status"] == "not_strict_spd"
                for record in records
            ),
            "runtime_error": sum(
                record["finite_precision_cone_status"] == "runtime_error"
                for record in records
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit()
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()

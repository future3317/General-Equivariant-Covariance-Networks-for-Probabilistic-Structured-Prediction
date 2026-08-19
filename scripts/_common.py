"""Shared command-line contracts for training and profiling entry points."""

from __future__ import annotations

import argparse

from equivcompiler import (
    AsinhExponentialCovariance,
    CenteredSpectralWindowCovariance,
    FirstFeasible,
    FullCovariance,
    GraphPrecision,
    IsotypicBlockCovariance,
    LowRankCovariance,
)
from models.backbone import CUEQ_METHODS, TENSOR_PRODUCT_BACKENDS
from representations import EquivariantOutputGraph


def add_tensor_product_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the tensor-product backend options used by every model script."""
    parser.add_argument(
        "--tp_backend",
        choices=TENSOR_PRODUCT_BACKENDS,
        default="e3nn",
        help="tensor-product implementation",
    )
    parser.add_argument(
        "--cueq_method",
        choices=CUEQ_METHODS,
        default="naive",
        help="cuEquivariance kernel; only used with --tp_backend cueq",
    )
    parser.add_argument(
        "--compile_tp",
        action="store_true",
        help="compile each edge tensor product with torch.compile(fullgraph=True)",
    )


def tensor_product_kwargs(args: argparse.Namespace) -> dict[str, str]:
    """Translate parsed CLI options to ``EquivariantBackbone`` arguments."""
    return {
        "tp_backend": args.tp_backend,
        "cueq_method": args.cueq_method,
    }


def covariance_policy_from_cli(
    name: str,
    *,
    rank: int,
    parameter_budget: int,
    graph: EquivariantOutputGraph | None = None,
    shape_min: float = -2.0,
    shape_max: float = 2.0,
    volume_min: float = -8.0,
    volume_max: float = 8.0,
):
    """Translate an explicit CLI choice to one typed covariance policy."""
    policies = {
        "full": FullCovariance(),
        "asinh_exponential": AsinhExponentialCovariance(),
        "low_rank": LowRankCovariance(rank),
        "block": IsotypicBlockCovariance(),
        "centered_spectral_window": CenteredSpectralWindowCovariance(
            shape_min=shape_min,
            shape_max=shape_max,
            volume_min=volume_min,
            volume_max=volume_max,
        ),
    }
    if graph is not None:
        policies["graph"] = GraphPrecision(graph)
    if name == "auto":
        priority = tuple(
            policies[item]
            for item in ("full", "graph", "low_rank", "block")
            if item in policies
        )
        return FirstFeasible(parameter_budget, priority)
    try:
        return policies[name]
    except KeyError as error:
        raise ValueError(f"unsupported covariance policy: {name}") from error

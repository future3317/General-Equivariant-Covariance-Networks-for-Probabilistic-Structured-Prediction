"""Benchmark equivalent e3nn and cuEquivariance tensor-product kernels.

This deliberately measures only the tensor-product operator.  Backbone graph
assembly, pooling, and the probabilistic readout are unchanged and should be
benchmarked separately.  The JSON record includes the complete shape/device
contract so timings are not reused across incompatible executions.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compatibility.e3nn import o3
from models.backbone import EquivariantMessagePassing


def _measure_forward(layer, inputs, edges, weights, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        layer.tp(inputs, edges, weights)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        layer.tp(inputs, edges, weights)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def _measure_forward_backward(
    layer, inputs, edges, weights, warmup: int, iters: int
) -> float:
    for _ in range(warmup):
        x = inputs.detach().requires_grad_(True)
        e = edges.detach().requires_grad_(True)
        w = weights.detach().requires_grad_(True)
        layer.tp(x, e, w).square().mean().backward()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        x = inputs.detach().requires_grad_(True)
        e = edges.detach().requires_grad_(True)
        w = weights.detach().requires_grad_(True)
        layer.tp(x, e, w).square().mean().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device(args.device)
    irreps_in = o3.Irreps("3x0e + 2x1o")
    irreps_out = o3.Irreps("4x0e + 2x1o")
    edge_irreps = o3.Irreps("0e + 1o")
    reference = EquivariantMessagePassing(
        irreps_in, irreps_out, edge_irreps, num_basis=4, tp_backend="e3nn"
    ).to(device)
    inputs = irreps_in.randn(args.edges, -1, device=device)
    edges = edge_irreps.randn(args.edges, -1, device=device)
    weights = torch.randn(
        args.edges, reference.tp.weight_numel, device=device
    )

    records = []
    # cuEquivariance may print a one-time GPU capability probe during its first
    # construction.  Keep the benchmark artifact machine-readable.
    with contextlib.redirect_stdout(io.StringIO()):
        for backend, method in (
            ("e3nn", "naive"),
            ("cueq", "naive"),
            ("cueq", "fused_tp"),
        ):
            if (
                backend == "cueq"
                and importlib.util.find_spec("cuequivariance_ops_torch") is None
                and method == "fused_tp"
            ):
                continue
            layer = EquivariantMessagePassing(
                irreps_in,
                irreps_out,
                edge_irreps,
                num_basis=4,
                tp_backend=backend,
                cueq_method=method,
            ).to(device)
            if layer.tp.weight_numel != reference.tp.weight_numel:
                raise RuntimeError("backend weight layouts are not equivalent")
            output = layer.tp(inputs, edges, weights)
            reference_output = reference.tp(inputs, edges, weights)
            error = (output - reference_output).abs().max().item()
            records.append(
                {
                    "backend": backend,
                    "method": method,
                    "max_abs_error": error,
                    "forward_ms": _measure_forward(
                        layer, inputs, edges, weights, args.warmup, args.iters
                    ),
                    "forward_backward_ms": _measure_forward_backward(
                        layer, inputs, edges, weights, args.warmup, args.iters
                    ),
                }
            )

    reference_forward = records[0]["forward_ms"]
    reference_backward = records[0]["forward_backward_ms"]
    for record in records:
        record["forward_speedup_vs_e3nn"] = reference_forward / record["forward_ms"]
        record["forward_backward_speedup_vs_e3nn"] = (
            reference_backward / record["forward_backward_ms"]
        )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "edges": args.edges,
                "irreps_in": str(irreps_in),
                "irreps_edge": str(edge_irreps),
                "irreps_out": str(irreps_out),
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

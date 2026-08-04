"""Measure the compiler system contract on a controlled readout workload.

This is deliberately a no-training benchmark.  It measures planning/material-
ization time and steady-state readout forward and forward/backward latency for
the same semantic plan.  Unsupported combinations are recorded as explicit
``unsupported`` rows; they are never replaced by another backend or dtype.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from equivcompiler import (
    ExactOnly,
    FeatureSpec,
    FullCovariance,
    PreferExecutor,
    SpecificExecutor,
    plan_readout,
)
from scripts.benchmarking import environment_record, measure, synchronize


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }[name]


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item < 1 for item in result):
        raise ValueError(f"expected positive comma-separated integers, got {value!r}")
    return result


def _build_case(
    *,
    multiplicity: int,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
):
    seed = FeatureSpec.from_irreps(
        f"{2 * multiplicity}x0e + {multiplicity}x1o + {multiplicity}x2e",
        scope="global",
    )
    started = time.perf_counter()
    plan = plan_readout(
        seed,
        output="ij=ji",
        covariance=FullCovariance(),
        fidelity=ExactOnly(),
        executor=SpecificExecutor(backend),
        cost=PreferExecutor((backend,)),
    )
    module = plan.build_readout(device=device, dtype=dtype)
    compile_ms = (time.perf_counter() - started) * 1000.0
    features = (0.1 * seed.irreps.randn(batch_size, -1, device=device, dtype=dtype)).requires_grad_()
    target = torch.randn(batch_size, 6, device=device, dtype=dtype)
    return plan, module, features, target, compile_ms


def _benchmark_module(
    module: torch.nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict:
    latest: dict | None = None

    def forward() -> None:
        nonlocal latest
        latest = module(features, target=target, return_scale=True)

    def forward_backward() -> None:
        forward()
        assert latest is not None
        latest["loss"].backward()

    def prepare() -> None:
        module.zero_grad(set_to_none=True)
        features.grad = None

    return {
        "forward": measure(
            forward,
            prepare=None,
            device=device,
            warmup=warmup,
            repeats=repeats,
        ),
        "forward_backward": measure(
            forward_backward,
            prepare=prepare,
            device=device,
            warmup=warmup,
            repeats=repeats,
        ),
    }


def _run_one(args: argparse.Namespace, mode: str, backend: str, dtype_name: str, batch: int, mult: int) -> dict:
    device = torch.device(args.device)
    dtype = _dtype(dtype_name)
    row = {
        "mode": mode,
        "backend": backend,
        "dtype": dtype_name,
        "batch_size": batch,
        "seed_multiplicity": mult,
        "status": "unsupported",
    }
    try:
        plan, module, features, target, compile_ms = _build_case(
            multiplicity=mult,
            backend=backend,
            device=device,
            dtype=dtype,
            batch_size=batch,
        )
        timed = module
        cold_compile_ms = None
        if mode == "compiled":
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is unavailable in this PyTorch build")
            timed = torch.compile(module)
            synchronize(device)
            started = time.perf_counter()
            timed(features, target=target, return_scale=True)
            synchronize(device)
            cold_compile_ms = (time.perf_counter() - started) * 1000.0
        timings = _benchmark_module(
            timed,
            features,
            target,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        row.update(
            {
                "status": "ok",
                "compile_ms": compile_ms,
                "cold_compile_ms": cold_compile_ms,
                "plan": plan.report.as_dict(),
                "timings": timings,
            }
        )
    except Exception as error:  # noqa: BLE001 - record every unsupported benchmark case
        row.update({"error_type": type(error).__name__, "error": str(error)})
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backends", default="spherical_cg,cartesian_stf")
    parser.add_argument("--dtypes", default="float32,bfloat16")
    parser.add_argument("--executions", default="eager,compiled")
    parser.add_argument("--batch-sizes", default="16,64,256")
    parser.add_argument("--multiplicities", default="8,16,32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 4:
        parser.error("--repeats must be at least 4")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    rows = []
    for mode in (item.strip() for item in args.executions.split(",")):
        for backend in (item.strip() for item in args.backends.split(",")):
            for dtype in (item.strip() for item in args.dtypes.split(",")):
                for batch in _parse_ints(args.batch_sizes):
                    for mult in _parse_ints(args.multiplicities):
                        rows.append(_run_one(args, mode, backend, dtype, batch, mult))
    result = {
        "kind": "compiler_system_benchmark",
        "training_steps": 0,
        "environment": environment_record(device),
        "protocol": {
            "unit": "one global V=0e+2e readout batch",
            "compile_time": "plan construction plus executable materialization",
            "latency": "synchronized steady-state wall time; data transfer excluded",
            "repeats": args.repeats,
            "warmup": args.warmup,
            "unsupported_policy": "record explicit status/error; no fallback",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

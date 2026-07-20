"""Scale sweep for exact spherical-CG versus dense-projector lowering.

This is a head-only benchmark; it never loads a dataset or starts training.
Each pair has identical compiled topology and checkpoint coordinates.  The
only change is the one-edge tensor-product execution schedule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from compatibility.e3nn import o3

from representations import CompilerConfig, O3RepresentationCompiler
from scripts.benchmarking import environment_record, measure


def _csv_ints(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",") if item.strip()]
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def _csv_choices(value: str, allowed: set[str]) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    invalid = set(parsed) - allowed
    if not parsed or invalid:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated values from {sorted(allowed)}, got {sorted(invalid)}"
        )
    return parsed


def _build_pair(
    seed: o3.Irreps,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    common = {"covariance": "full", "output_scope": "dense", "objective": "gaussian"}
    spherical = O3RepresentationCompiler(
        "0e + 2e", CompilerConfig(**common, backend="spherical_cg")
    ).compile(seed).build_head()
    lowered = O3RepresentationCompiler(
        "0e + 2e", CompilerConfig(**common, backend="cartesian_stf")
    ).compile(seed).build_head()
    lowered.load_state_dict(spherical.state_dict(), strict=True)
    return (
        spherical.to(device=device, dtype=dtype),
        lowered.to(device=device, dtype=dtype),
    )


def _compile(module: torch.nn.Module, mode: str) -> torch.nn.Module:
    if mode == "eager":
        return module
    return torch.compile(module, dynamic=False, fullgraph=False)


def _benchmark_one(
    *,
    batch_size: int,
    multiplicity: int,
    dtype_name: str,
    execution: str,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict:
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    seed = o3.Irreps(
        f"{2 * multiplicity}x0e + {multiplicity}x1o + {multiplicity}x2e"
    )
    spherical, lowered = _build_pair(seed, device=device, dtype=dtype)
    spherical = _compile(spherical, execution)
    lowered = _compile(lowered, execution)
    features = (0.1 * seed.randn(batch_size, -1, device=device, dtype=dtype)).requires_grad_()

    with torch.no_grad():
        spherical_mean, spherical_operator = spherical(features)
        lowered_mean, lowered_operator = lowered(features)
        maximum_mean_error = (spherical_mean - lowered_mean).abs().max()
        maximum_operator_error = (spherical_operator - lowered_operator).abs().max()

    result = {
        "batch_size": batch_size,
        "multiplicity_2e": multiplicity,
        "seed_irreps": str(seed),
        "dtype": dtype_name,
        "execution": execution,
        "maximum_mean_error": float(maximum_mean_error),
        "maximum_operator_error": float(maximum_operator_error),
        "spherical_cg": {},
        "dense_projector": {},
    }
    for name, head in (("spherical_cg", spherical), ("dense_projector", lowered)):
        latest: tuple[torch.Tensor, torch.Tensor] | None = None

        def forward() -> None:
            nonlocal latest
            latest = head(features)

        def forward_backward() -> None:
            nonlocal latest
            latest = head(features)
            mean, operator = latest
            (mean.square().mean() + operator.square().mean()).backward()

        def prepare() -> None:
            head.zero_grad(set_to_none=True)
            features.grad = None

        result[name]["forward"] = measure(
            forward,
            prepare=None,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        result[name]["forward_backward"] = measure(
            forward_backward,
            prepare=prepare,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
    for phase in ("forward", "forward_backward"):
        baseline = result["spherical_cg"][phase]["median_ms"]
        optimized = result["dense_projector"][phase]["median_ms"]
        result["dense_projector"][phase]["speedup"] = baseline / optimized
    return result


def benchmark(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    records = []
    for batch_size in args.batch_sizes:
        for multiplicity in args.multiplicities:
            for dtype_name in args.dtypes:
                if dtype_name == "bfloat16" and device.type != "cuda":
                    records.append(
                        {
                            "batch_size": batch_size,
                            "multiplicity_2e": multiplicity,
                            "dtype": dtype_name,
                            "status": "unsupported",
                            "reason": "BF16 is benchmarked only on CUDA",
                        }
                    )
                    continue
                for execution in args.executions:
                    try:
                        record = _benchmark_one(
                            batch_size=batch_size,
                            multiplicity=multiplicity,
                            dtype_name=dtype_name,
                            execution=execution,
                            device=device,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )
                        record["status"] = "ok"
                    except Exception as error:
                        record = {
                            "batch_size": batch_size,
                            "multiplicity_2e": multiplicity,
                            "dtype": dtype_name,
                            "execution": execution,
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    records.append(record)
    return {
        "kind": "head_microbenchmark_scale_sweep",
        "environment": environment_record(device),
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "baseline": {
                "module": "e3nn.o3.FullyConnectedTensorProduct",
                "internal_weights": True,
                "shared_weights": True,
                "irrep_normalization": "component",
                "path_normalization": "element",
            },
            "lowering": "same topology, flat weights, path normalization, and output basis",
        },
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", type=_csv_ints, default=_csv_ints("32,128,256"))
    parser.add_argument("--multiplicities", type=_csv_ints, default=_csv_ints("8,16,32,64"))
    parser.add_argument(
        "--dtypes",
        type=lambda value: _csv_choices(value, {"float32", "bfloat16"}),
        default=["float32"],
    )
    parser.add_argument(
        "--executions",
        type=lambda value: _csv_choices(value, {"eager", "compile"}),
        default=["eager", "compile"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = benchmark(args)
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

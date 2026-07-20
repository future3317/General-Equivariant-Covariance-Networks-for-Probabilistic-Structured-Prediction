"""Shared, auditable timing helpers for non-training benchmarks."""

from __future__ import annotations

import importlib.metadata
import platform
import statistics
import time
from collections.abc import Callable

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_milliseconds(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "median_ms": statistics.median(ordered),
        "iqr_ms": quartiles[2] - quartiles[0],
        "q1_ms": quartiles[0],
        "q3_ms": quartiles[2],
        "mean_ms": statistics.fmean(ordered),
        "std_ms": statistics.pstdev(ordered),
        "repeats": len(ordered),
    }


def measure(
    run: Callable[[], None],
    *,
    prepare: Callable[[], None] | None,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    """Measure ``run`` with synchronization and untimed preparation.

    In backward benchmarks ``prepare`` performs ``zero_grad(set_to_none=True)``
    before synchronization and before the timer.  Data transfer and graph
    construction are likewise expected to happen outside this function.
    """
    if repeats < 4:
        raise ValueError("repeats must be at least 4 to report an IQR")
    for _ in range(warmup):
        if prepare is not None:
            prepare()
        run()
        synchronize(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(repeats):
        if prepare is not None:
            prepare()
        synchronize(device)
        start = time.perf_counter()
        run()
        synchronize(device)
        timings.append(1000.0 * (time.perf_counter() - start))
    result = summarize_milliseconds(timings)
    if device.type == "cuda":
        result.update(
            {
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20,
            }
        )
    return result


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_record(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "e3nn": _version("e3nn"),
        "cuequivariance": _version("cuequivariance"),
        "cuequivariance_torch": _version("cuequivariance-torch"),
        "triton": _version("triton"),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "tf32_matmul": (
            bool(torch.backends.cuda.matmul.allow_tf32)
            if device.type == "cuda"
            else None
        ),
        "cudnn_benchmark": (
            bool(torch.backends.cudnn.benchmark) if device.type == "cuda" else None
        ),
        "timing": {
            "synchronize_before_and_after": device.type == "cuda",
            "zero_grad": "set_to_none=True, before timer",
            "data_transfer": "excluded",
            "memory": (
                "absolute process peak allocated and reserved; cache emptied and "
                "peak counters reset after warmup for each phase"
            ),
        },
    }

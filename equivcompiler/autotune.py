"""Measured, shape-specific executor autotuning utilities."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Mapping

import torch

from equivcompiler.policies import ExecutorMeasurement, ShapeSignature


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


@dataclass(frozen=True)
class BenchmarkTask:
    """Timed callable with untimed state preparation."""

    run: Callable[[], object]
    prepare: Callable[[], None] | None = None


@dataclass(frozen=True)
class DeviceAutotuner:
    """Benchmark already-lowered exact candidates for one shape signature."""

    warmup: int = 10
    iterations: int = 30

    def __post_init__(self) -> None:
        if self.warmup < 0 or self.iterations < 1:
            raise ValueError("warmup must be nonnegative and iterations positive")

    def measure(
        self,
        executor: str,
        signature: ShapeSignature,
        run_once: Callable[[], object],
        prepare: Callable[[], None] | None = None,
    ) -> ExecutorMeasurement:
        for _ in range(self.warmup):
            if prepare is not None:
                prepare()
            run_once()
        _synchronize(signature.device)
        samples = []
        for _ in range(self.iterations):
            if prepare is not None:
                prepare()
            _synchronize(signature.device)
            start = time.perf_counter()
            run_once()
            _synchronize(signature.device)
            samples.append((time.perf_counter() - start) * 1_000.0)
        values = torch.tensor(samples, dtype=torch.float64)
        q1, median, q3 = torch.quantile(
            values, values.new_tensor([0.25, 0.5, 0.75])
        )
        return ExecutorMeasurement(
            executor=executor,  # type: ignore[arg-type]
            signature=signature,
            median_ms=float(median),
            iqr_ms=float(q3 - q1),
        )

    def benchmark(
        self,
        signature: ShapeSignature,
        runners: Mapping[str, Callable[[], object] | BenchmarkTask],
    ) -> tuple[ExecutorMeasurement, ...]:
        return tuple(
            self.measure(
                name,
                signature,
                task.run if isinstance(task, BenchmarkTask) else task,
                task.prepare if isinstance(task, BenchmarkTask) else None,
            )
            for name, task in runners.items()
        )

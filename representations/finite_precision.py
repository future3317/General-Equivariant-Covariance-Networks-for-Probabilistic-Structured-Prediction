"""Runtime certificates for the finite-precision SPD cone."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NumericalSPDCertificate:
    """Value-dependent SPD certificate for one materialized batch of matrices.

    Mathematical SPD is established by the typed IR.  This certificate is a
    separate runtime check: it rejects non-finite or numerically marginal
    matrices relative to the requested dtype and an explicit scale-aware
    tolerance.  No jitter or silent repair is applied.
    """

    dtype: str
    finite: bool
    minimum_eigenvalue: float | None
    scale: float | None
    threshold: float | None
    strict: bool
    policy: str = "reject_if_not_strict"

    def as_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "finite": self.finite,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "scale": self.scale,
            "threshold": self.threshold,
            "strict": self.strict,
            "policy": self.policy,
        }


def certify_numerical_spd(
    matrix: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    margin: float = 16.0,
    minimum_eigenvalue: float | None = None,
) -> NumericalSPDCertificate:
    """Certify a materialized symmetric matrix in its execution envelope.

    Eigenvalues are inspected in float64 to avoid making the audit itself
    less precise than the matrix under test.  The threshold scales with the
    largest absolute matrix entry and the machine epsilon of the declared
    execution dtype.  Callers must reject a failed certificate; this helper
    intentionally does not add jitter or change the matrix.
    """
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must have square trailing dimensions")
    declared_dtype = matrix.dtype if dtype is None else dtype
    if declared_dtype not in {
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
    }:
        raise TypeError(f"unsupported numerical SPD dtype: {declared_dtype}")
    work = matrix.detach().to(torch.float64)
    finite = bool(torch.isfinite(work).all())
    if not finite:
        return NumericalSPDCertificate(
            dtype=str(declared_dtype).removeprefix("torch."),
            finite=False,
            minimum_eigenvalue=None,
            scale=None,
            threshold=None,
            strict=False,
        )
    symmetric = 0.5 * (work + work.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    minimum = float(eigenvalues.min().item())
    scale = float(work.abs().amax().item())
    if minimum_eigenvalue is None:
        epsilon = float(torch.finfo(declared_dtype).eps)
        threshold = margin * epsilon * max(1.0, scale)
    else:
        if minimum_eigenvalue < 0.0:
            raise ValueError("minimum_eigenvalue threshold must be nonnegative")
        threshold = float(minimum_eigenvalue)
    return NumericalSPDCertificate(
        dtype=str(declared_dtype).removeprefix("torch."),
        finite=True,
        minimum_eigenvalue=minimum,
        scale=scale,
        threshold=threshold,
        strict=minimum > threshold,
    )

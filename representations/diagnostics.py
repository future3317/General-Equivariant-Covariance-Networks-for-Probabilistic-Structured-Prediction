"""Structured diagnostics emitted by the representation compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


CertificateStatus = Literal[
    "success",
    "failure",
    "approximation",
    "restriction",
    "diagnostic",
]


@dataclass(frozen=True)
class CompilationCertificate:
    """Machine-readable proof, safeguard, or approximation record."""

    code: str
    status: CertificateStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "status": self.status,
            "message": self.message,
            "details": dict(self.details),
        }


class CompilationError(ValueError):
    """Compilation failure carrying a stable, machine-readable certificate."""

    def __init__(self, certificate: CompilationCertificate):
        if certificate.status != "failure":
            raise ValueError("CompilationError requires a failure certificate")
        self.certificate = certificate
        super().__init__(certificate.message)

    def as_dict(self) -> dict[str, Any]:
        return self.certificate.as_dict()


class UnreachableTargetError(CompilationError):
    """A canonical or active target cannot be generated from the seed."""


class UnreachableActiveTargetError(UnreachableTargetError):
    """The selected statistical family's active parameter target is unreachable."""

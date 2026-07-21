"""Exact execution signatures for measured backend selection."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from typing import TYPE_CHECKING, Any

import torch

from equivcompiler.policies import ExecutionSignature

if TYPE_CHECKING:
    from equivcompiler.planning import CompilationPlan


def _hash(record: Any) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def software_fingerprint() -> str:
    packages = {}
    for name in ("torch", "e3nn", "cuequivariance", "triton"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    packages["torch_cuda_build"] = torch.version.cuda
    return _hash(packages)


def device_fingerprint(device: torch.device | str) -> str:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return _hash({"type": resolved.type})
    index = (
        resolved.index if resolved.index is not None else torch.cuda.current_device()
    )
    properties = torch.cuda.get_device_properties(index)
    record = {
        "name": properties.name,
        "total_memory": properties.total_memory,
        "capability": torch.cuda.get_device_capability(index),
        "uuid": str(getattr(properties, "uuid", "unavailable")),
    }
    return _hash(record)


def plan_fingerprints(
    *,
    feature_record: dict,
    output_record: dict,
    active_plan_record: dict,
    operator_record: dict,
    distribution_record: dict,
) -> dict[str, str]:
    active_plan_hash = _hash(active_plan_record)
    operator_program_hash = _hash(operator_record)
    semantic_plan_hash = _hash(
        {
            "feature": feature_record,
            "output": output_record,
            "active_plan_hash": active_plan_hash,
            "operator_program_hash": operator_program_hash,
            "distribution": distribution_record,
        }
    )
    return {
        "active_plan_hash": active_plan_hash,
        "operator_program_hash": operator_program_hash,
        "semantic_plan_hash": semantic_plan_hash,
    }


def execution_signature_for_plan(
    plan: "CompilationPlan",
    *,
    batch_shape: tuple[int, ...],
    dtype: str,
    device: torch.device | str,
    phase: str = "forward_backward",
    compilation_mode: str = "eager",
    tensor_layout: str = "contiguous",
    requires_input_grad: bool = True,
    requires_parameter_grad: bool = True,
) -> ExecutionSignature:
    fingerprints = plan_fingerprints(
        feature_record=plan.seed.as_dict(),
        output_record=plan.semantics.as_dict(),
        active_plan_record=plan.compilation.active_plan.as_dict(),
        operator_record=plan.compilation.operator_family.assembly.semantic_dict(),
        distribution_record=plan.distribution_spec.as_dict(),
    )
    return ExecutionSignature(
        semantic_plan_hash=fingerprints["semantic_plan_hash"],
        feature_fingerprint=plan.seed.fingerprint,
        active_plan_hash=fingerprints["active_plan_hash"],
        operator_program_hash=fingerprints["operator_program_hash"],
        batch_shape=batch_shape,
        dtype=dtype,
        device=str(device),
        device_uuid=device_fingerprint(device),
        phase=phase,  # type: ignore[arg-type]
        compilation_mode=compilation_mode,  # type: ignore[arg-type]
        software_fingerprint=software_fingerprint(),
        tensor_layout=tensor_layout,
        requires_input_grad=requires_input_grad,
        requires_parameter_grad=requires_parameter_grad,
    )

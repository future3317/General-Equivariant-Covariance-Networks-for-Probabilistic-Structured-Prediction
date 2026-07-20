"""Strict checkpoint migration between algebraically equivalent backends."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
from compatibility.e3nn import o3

from equivcompiler.api import normalize_backend
from equivcompiler.planning import plan_readout
from equivcompiler.policies import (
    ExactOnly,
    FullCovariance,
    IsotypicBlockCovariance,
    LowRankCovariance,
)
from equivcompiler.specs import FeatureSpec
from representations import (
    CompilationCertificate,
    CompilationError,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_container(payload: Any) -> tuple[dict[str, torch.Tensor], str | None]:
    if not isinstance(payload, Mapping):
        raise CompilationError(
            CompilationCertificate(
                code="invalid_checkpoint",
                status="failure",
                message="checkpoint must be a state dictionary or contain one",
            )
        )
    for key in ("state_dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in candidate.items()
        ):
            return dict(candidate), key
    if all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in payload.items()
    ):
        return dict(payload), None
    raise CompilationError(
        CompilationCertificate(
            code="invalid_checkpoint",
            status="failure",
            message="no tensor state dictionary was found in the checkpoint",
            details={"top_level_keys": [str(key) for key in payload]},
        )
    )


def _head_prefix(state: Mapping[str, torch.Tensor], expected: set[str]) -> str:
    candidates = ("joint_head.", "module.joint_head.", "head.", "module.head.", "")
    learned_expected = {key for key in expected if not key.endswith("output_mask")}
    for prefix in candidates:
        present = {key[len(prefix) :] for key in state if key.startswith(prefix)}
        if learned_expected <= present:
            return prefix
    raise CompilationError(
        CompilationCertificate(
            code="checkpoint_layout_mismatch",
            status="failure",
            message="checkpoint does not contain the compiled readout coordinate set",
            details={"expected_head_keys": sorted(learned_expected)},
        )
    )


def _extract_head_state(
    state: Mapping[str, torch.Tensor], prefix: str, expected: set[str]
) -> dict[str, torch.Tensor]:
    generated = "_compiled_main_left_right."
    extracted = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            continue
        local = key[len(prefix) :]
        if local in expected or local.endswith("output_mask") or generated in local:
            extracted[local] = value
    return extracted


def _deterministic_backend_buffer(key: str, head_prefix: str) -> bool:
    if not key.startswith(head_prefix):
        return False
    local = key[len(head_prefix) :]
    return local.endswith("tensor_product.output_mask") or (
        ".tensor_product._compiled_main_left_right." in local
    )


def _compiler(
    output: str,
    seed_irreps: o3.Irreps,
    backend: str,
    *,
    covariance: str,
    feature_scope: str,
    output_scope: str,
    distribution: str,
    budget: int,
    low_rank: int,
) -> tuple[Any, Any, Any]:
    del budget
    if covariance == "full":
        family = FullCovariance()
    elif covariance == "low_rank":
        family = LowRankCovariance(low_rank)
    elif covariance == "block":
        family = IsotypicBlockCovariance()
    else:
        raise CompilationError(
            CompilationCertificate(
                code="unsupported_checkpoint_family",
                status="failure",
                message="checkpoint migration currently requires full, low_rank, or block covariance",
                details={"requested": covariance},
            )
        )
    seed = FeatureSpec.from_irreps(seed_irreps, scope=feature_scope)
    semantic_output_scope = feature_scope if output_scope == "dense" else output_scope
    plan = plan_readout(
        seed,
        output=output,
        covariance=family,
        lowering=ExactOnly(backend=backend),
        distribution=distribution,
        output_scope=semantic_output_scope,
    )
    return plan, plan.compilation, plan.compilation.build_head()


def convert_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    from_backend: str,
    to_backend: str,
    output: str,
    seed_irreps: str | o3.Irreps,
    covariance: str = "full",
    feature_scope: str = "node",
    output_scope: str = "global",
    distribution: str = "gaussian",
    budget: int = 192,
    low_rank: int = 8,
    audit_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and migrate a checkpoint without changing learned coordinates."""
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.resolve() == destination_path.resolve():
        raise CompilationError(
            CompilationCertificate(
                code="in_place_checkpoint_conversion",
                status="failure",
                message="source and destination must differ; in-place conversion is refused",
            )
        )
    if destination_path.exists() and not overwrite:
        raise FileExistsError(
            f"destination exists: {destination_path}; pass overwrite=True explicitly"
        )
    source_backend = normalize_backend(from_backend)
    target_backend = normalize_backend(to_backend)
    if source_backend == "auto" or target_backend == "auto":
        raise CompilationError(
            CompilationCertificate(
                code="ambiguous_checkpoint_backend",
                status="failure",
                message="checkpoint conversion requires explicit source and target backends",
            )
        )

    seed = o3.Irreps(seed_irreps)
    source_plan, source_compilation, source_head = _compiler(
        output,
        seed,
        source_backend,
        covariance=covariance,
        feature_scope=feature_scope,
        output_scope=output_scope,
        distribution=distribution,
        budget=budget,
        low_rank=low_rank,
    )
    target_plan, target_compilation, target_head = _compiler(
        output,
        seed,
        target_backend,
        covariance=covariance,
        feature_scope=feature_scope,
        output_scope=output_scope,
        distribution=distribution,
        budget=budget,
        low_rank=low_rank,
    )
    if not source_compilation.backend_exact or not target_compilation.backend_exact:
        raise CompilationError(
            CompilationCertificate(
                code="non_exact_checkpoint_conversion",
                status="failure",
                message="checkpoint conversion refuses approximate contraction backends",
            )
        )
    compatibility_fields = (
        "canonical_target_irreps",
        "active_target_irreps",
        "covariance_mode",
        "covariance_rank",
    )
    mismatches = {
        field: [str(getattr(source_compilation, field)), str(getattr(target_compilation, field))]
        for field in compatibility_fields
        if getattr(source_compilation, field) != getattr(target_compilation, field)
    }
    if mismatches:
        raise CompilationError(
            CompilationCertificate(
                code="incompatible_compilations",
                status="failure",
                message="source and target compilations do not have identical semantics",
                details={"mismatches": mismatches},
            )
        )

    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    state, container_key = _state_container(payload)
    source_expected = set(source_head.state_dict())
    prefix = _head_prefix(state, source_expected)
    source_head_state = _extract_head_state(state, prefix, source_expected)
    source_head.load_state_dict(deepcopy(source_head_state), strict=True)
    target_head.load_state_dict(deepcopy(source_head_state), strict=True)

    source_parameters = dict(source_head.named_parameters())
    target_parameters = dict(target_head.named_parameters())
    if source_parameters.keys() != target_parameters.keys() or any(
        source_parameters[name].shape != target_parameters[name].shape
        for name in source_parameters
    ):
        raise CompilationError(
            CompilationCertificate(
                code="checkpoint_coordinate_mismatch",
                status="failure",
                message="learned parameter names or shapes changed across backends",
            )
        )

    generator = torch.Generator(device="cpu").manual_seed(20260720)
    features = torch.randn(4, seed.dim, generator=generator)
    batch = torch.zeros(4, dtype=torch.long) if output_scope == "global" else None
    source_head.eval()
    target_head.eval()
    with torch.inference_mode():
        source_mean, source_parameters_out = source_head(features, batch)
        target_mean, target_parameters_out = target_head(features, batch)
    equivalence = {
        "mean_max_abs": float((source_mean - target_mean).abs().max()),
        "distribution_parameter_max_abs": float(
            (source_parameters_out - target_parameters_out).abs().max()
        ),
    }

    removed = [
        key for key in state if _deterministic_backend_buffer(key, prefix)
    ]
    converted_state = {
        key: value for key, value in state.items() if key not in set(removed)
    }
    if container_key is None:
        converted_payload: Any = converted_state
    else:
        converted_payload = dict(payload)
        converted_payload[container_key] = converted_state

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted_payload, destination_path)
    audit = {
        "schema_version": 1,
        "operation": "exact_checkpoint_backend_migration",
        "source": {
            "path": str(source_path.resolve()),
            "sha256": _sha256(source_path),
            "backend": source_backend,
        },
        "destination": {
            "path": str(destination_path.resolve()),
            "sha256": _sha256(destination_path),
            "backend": target_backend,
        },
        "state_container": container_key or "raw_state_dict",
        "head_prefix": prefix,
        "learned_coordinates_changed": False,
        "learned_tensor_count": len(source_parameters),
        "removed_deterministic_buffers": removed,
        "numerical_equivalence": equivalence,
        "source_compatibility_hash": source_plan.compatibility_hash,
        "target_compatibility_hash": target_plan.compatibility_hash,
        "feature_fingerprint": source_plan.seed.fingerprint,
        "source_compilation": source_plan.report.as_dict(),
        "target_compilation": target_plan.report.as_dict(),
    }
    audit_file = (
        Path(audit_path)
        if audit_path is not None
        else destination_path.with_suffix(destination_path.suffix + ".conversion.json")
    )
    if audit_file.exists() and not overwrite:
        destination_path.unlink(missing_ok=True)
        raise FileExistsError(
            f"audit destination exists: {audit_file}; pass overwrite=True explicitly"
        )
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    audit["audit_path"] = str(audit_file.resolve())
    return audit

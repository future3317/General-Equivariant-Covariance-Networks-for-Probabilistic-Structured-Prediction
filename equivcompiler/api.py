"""Unambiguous public entry points for readout and predictor compilation."""

from __future__ import annotations

from typing import Literal

import torch

from equivcompiler.planning import CompilationPlan, plan_readout
from equivcompiler.policies import (
    CostPolicy,
    CovariancePolicy,
    ExactExecutorCandidates,
    ExecutorPolicy,
    FidelityPolicy,
    FullCovariance,
    PreferExecutor,
)
from equivcompiler.distributions import DistributionSpec
from equivcompiler.specs import FeatureSpec
from representations import CompilationReport


def normalize_backend(backend: str) -> str:
    """Normalize CLI spelling for compatibility with the internal compiler."""
    aliases = {
        "auto": "auto",
        "spherical_cg": "spherical_cg",
        "spherical-cg": "spherical_cg",
        "cartesian_stf": "cartesian_stf",
        "cartesian-stf": "cartesian_stf",
        "dense_projector": "cartesian_stf",
        "dense-projector": "cartesian_stf",
    }
    try:
        return aliases[backend.lower()]
    except KeyError as error:
        from representations import CompilationCertificate, CompilationError

        raise CompilationError(
            CompilationCertificate(
                code="unsupported_execution_backend",
                status="failure",
                message=f"unknown execution backend: {backend}",
                details={"requested": backend, "supported": sorted(aliases)},
            )
        ) from error


def compile_readout(
    seed: FeatureSpec,
    *,
    output,
    covariance: CovariancePolicy = FullCovariance(),
    distribution: DistributionSpec | Literal["gaussian", "student_t"] = "gaussian",
    fidelity: FidelityPolicy | None = None,
    executor: ExecutorPolicy = ExactExecutorCandidates(),
    cost: CostPolicy = PreferExecutor(),
    lowering: FidelityPolicy | None = None,
    student_t_dof: float = 5.0,
    output_scope: Literal["global", "node", "edge"] = "global",
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.nn.Module, CompilationReport]:
    """Compile a standalone probabilistic readout from a feature contract."""
    plan = plan_readout(
        seed,
        output=output,
        covariance=covariance,
        fidelity=fidelity,
        executor=executor,
        cost=cost,
        lowering=lowering,
        distribution=distribution,
        student_t_dof=student_t_dof,
        output_scope=output_scope,
    )
    readout = plan.build_readout(device=device, dtype=dtype)
    return readout, plan.report_for(readout)


def compile_predictor(
    backbone: torch.nn.Module,
    *,
    output,
    covariance: CovariancePolicy = FullCovariance(),
    distribution: DistributionSpec | Literal["gaussian", "student_t"] = "gaussian",
    fidelity: FidelityPolicy | None = None,
    executor: ExecutorPolicy = ExactExecutorCandidates(),
    cost: CostPolicy = PreferExecutor(),
    lowering: FidelityPolicy | None = None,
    student_t_dof: float = 5.0,
    output_scope: Literal["global", "node", "edge"] = "global",
) -> tuple[torch.nn.Module, CompilationReport]:
    """Compile and bind a complete predictor to a concrete backbone."""
    seed = FeatureSpec.from_backbone(backbone)
    plan = plan_readout(
        seed,
        output=output,
        covariance=covariance,
        fidelity=fidelity,
        executor=executor,
        cost=cost,
        lowering=lowering,
        distribution=distribution,
        student_t_dof=student_t_dof,
        output_scope=output_scope,
    )
    predictor = plan.bind(backbone)
    return predictor, plan.report_for(predictor)


__all__ = ["CompilationPlan", "compile_predictor", "compile_readout", "plan_readout"]

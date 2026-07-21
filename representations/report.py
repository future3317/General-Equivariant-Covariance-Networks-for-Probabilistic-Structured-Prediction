"""Auditable reports for compiled probabilistic output representations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from compatibility.e3nn import o3

from representations.adaptive_lifting import coverage_deficit, irrep_multiplicities
from representations.diagnostics import CompilationCertificate

if TYPE_CHECKING:
    from representations.compiler import O3Compilation


def _parity_name(parity: int) -> str:
    return "even" if parity == 1 else "odd"


def _irrep_record(irreps: o3.Irreps) -> dict[str, Any]:
    representation = o3.Irreps(irreps)
    multiplicities = irrep_multiplicities(representation)
    components = [
        {
            "irrep": str(irrep),
            "angular_momentum": irrep.l,
            "parity": _parity_name(irrep.p),
            "multiplicity": multiplicity,
            "dimension": multiplicity * irrep.dim,
        }
        for irrep, multiplicity in sorted(
            multiplicities.items(), key=lambda item: (item[0].l, -item[0].p)
        )
    ]
    return {
        "irreps": str(representation),
        "dimension": representation.dim,
        "highest_angular_momentum": max(
            (irrep.l for irrep in multiplicities), default=0
        ),
        "parities": sorted({_parity_name(irrep.p) for irrep in multiplicities}),
        "components": components,
    }


def _linear_parameter_count(irreps_in: o3.Irreps, irreps_out: o3.Irreps) -> int:
    return sum(
        left_mul * right_mul
        for left_mul, left_irrep in o3.Irreps(irreps_in)
        for right_mul, right_irrep in o3.Irreps(irreps_out)
        if left_irrep == right_irrep
    )


def _tensor_product_parameter_count(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    contraction_rank: int | None,
) -> int:
    count = 0
    for left_mul, left_irrep in o3.Irreps(irreps_in1):
        for right_mul, right_irrep in o3.Irreps(irreps_in2):
            allowed = set(left_irrep * right_irrep)
            for output_mul, output_irrep in o3.Irreps(irreps_out):
                if output_irrep not in allowed:
                    continue
                if contraction_rank is None:
                    count += left_mul * right_mul * output_mul
                else:
                    rank = min(contraction_rank, left_mul, right_mul)
                    count += output_mul * rank * (left_mul + right_mul)
    return count


def _readout_parameter_count(compilation: "O3Compilation") -> int:
    plan = compilation.active_plan
    count = 0
    if plan.depth == 0:
        count += _linear_parameter_count(plan.seed_irreps, plan.target_irreps)
    else:
        contraction_rank = (
            compilation.stf_contraction_rank
            if compilation.backend == "cartesian_stf"
            else None
        )
        for stage in plan.stages:
            count += _linear_parameter_count(stage.irreps_in, stage.irreps_out)
            count += _tensor_product_parameter_count(
                stage.irreps_in,
                plan.seed_irreps,
                stage.irreps_out,
                contraction_rank,
            )

    active = compilation.active_target_irreps
    count += _linear_parameter_count(active, compilation.mean_irreps)
    for binding in compilation.operator_family.parameter_bindings:
        count += _linear_parameter_count(active, binding.irreps)
    return count


def _covariance_complexity(compilation: "O3Compilation") -> dict[str, Any]:
    dimension = compilation.output_spec.dim
    mode = compilation.covariance_mode
    if mode == "full":
        return {
            "parameterization": "unrestricted_symmetric_generator",
            "emitted_coordinates": dimension * (dimension + 1) // 2,
            "storage": "O(d^2)",
            "likelihood_linear_algebra": "O(d^3)",
        }
    if mode == "block":
        multiplicities = irrep_multiplicities(compilation.mean_irreps)
        return {
            "parameterization": "isotypic_multiplicity_blocks",
            "emitted_coordinates": compilation.covariance_parameter_count,
            "storage": "O(sum_lambda m_lambda^2)",
            "likelihood_linear_algebra": "O(sum_lambda m_lambda^3)",
            "multiplicity_block_sizes": {
                str(irrep): multiplicity
                for irrep, multiplicity in multiplicities.items()
            },
        }
    if mode == "low_rank":
        rank = int(compilation.covariance_rank)
        return {
            "parameterization": "low_rank_plus_isotropic",
            "emitted_coordinates": dimension * rank + 1,
            "storage": "O(d r)",
            "likelihood_linear_algebra": "O(d r^2 + r^3)",
            "rank": rank,
        }
    if mode == "graph":
        assert compilation.graph_structure is not None
        graph = compilation.graph_structure
        block = graph.block_dim
        return {
            "parameterization": "local_spd_graph_precision",
            "emitted_coordinates": compilation.covariance_parameter_count,
            "storage": "O((J+E) d0^2)",
            "likelihood_linear_algebra": (
                "O(J d0^3 + E d0^2)" if graph.is_tree else "O((J d0)^3)"
            ),
            "nodes": graph.num_nodes,
            "edges": graph.num_edges,
            "block_dimension": block,
            "tree_elimination": graph.is_tree,
        }
    return {
        "parameterization": mode,
        "emitted_coordinates": compilation.covariance_parameter_count,
        "storage": "derived_from_operator_program",
        "likelihood_linear_algebra": "generic_dense_O(d^3)",
        "operator_program_hash": compilation.operator_family.assembly.fingerprint,
    }


def _probability_semantics(compilation: "O3Compilation") -> dict[str, Any]:
    specification = compilation.distribution_spec
    objective = specification.objective_name()
    if objective == "gaussian":
        scale_semantics = "covariance"
        covariance_relation = "covariance = scale"
        likelihood = {"name": "multivariate_gaussian", "proper": True}
    elif objective == "student_t":
        degrees = specification.student_t_dof
        assert degrees is not None
        scale_semantics = "scatter"
        covariance_relation = (
            f"covariance = {degrees}/({degrees}-2) * scatter"
            if degrees > 2
            else "covariance is undefined because degrees_of_freedom <= 2"
        )
        likelihood = {
            "name": "multivariate_student_t",
            "proper": True,
            "degrees_of_freedom": degrees,
        }
    else:
        scale_semantics = "distribution_defined_scatter"
        covariance_relation = "defined by radial-law plugin"
        likelihood = specification.objective()

    if compilation.covariance_mode == "graph":
        raw_semantics = "local symmetric precision generators"
        positive_definite_object = "global precision"
        precision_semantics = (
            "assembled directly from unary and relational SPD potentials"
        )
    else:
        raw_semantics = {
            "full": "symmetric scale generator",
            "block": "isotypic multiplicity-space generators",
            "low_rank": "equivariant factor and isotropic variance generator",
        }.get(
            compilation.covariance_mode,
            "typed coordinates consumed by the verified operator program",
        )
        positive_definite_object = scale_semantics
        precision_semantics = "inverse of the predicted scale matrix"
    return {
        "raw_parameter_semantics": raw_semantics,
        "positive_definite_object": positive_definite_object,
        "scale_semantics": scale_semantics,
        "precision_semantics": precision_semantics,
        "covariance_relation": covariance_relation,
        "likelihood": likelihood,
    }


@dataclass(frozen=True)
class CompilationReport:
    """Immutable, serializable report returned with every compiled executable."""

    _json: str

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "CompilationReport":
        return cls(json.dumps(record, sort_keys=True, separators=(",", ":")))

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def with_updates(self, **updates: Any) -> "CompilationReport":
        record = self.as_dict()
        record.update(updates)
        return self.from_dict(record)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @property
    def schema_version(self) -> str:
        return str(self["schema_version"])

    @property
    def output(self) -> dict[str, Any]:
        return self["output"]

    @property
    def seed(self) -> dict[str, Any]:
        return self["seed"]

    @property
    def targets(self) -> dict[str, Any]:
        return self["targets"]

    @property
    def representation_reachability(self) -> dict[str, Any]:
        return self["representation_reachability"]

    @property
    def family(self) -> dict[str, Any]:
        return self["family"]

    @property
    def execution_fidelity(self) -> dict[str, Any]:
        return self["execution_fidelity"]

    @property
    def backend_selection_basis(self) -> dict[str, Any]:
        return self["backend_selection_basis"]

    @property
    def objective(self) -> dict[str, Any]:
        return self["objective"]

    @property
    def complexity(self) -> dict[str, Any]:
        return self["complexity"]

    @property
    def compatibility_hash(self) -> str | None:
        return self["compatibility_hash"]


def build_compilation_report(
    compilation: "O3Compilation",
    executable: torch.nn.Module | None = None,
) -> CompilationReport:
    from representations.operator_lowering import match_optimized_program

    canonical_plan = compilation.canonical_plan
    canonical_deficit = (
        coverage_deficit(canonical_plan.irreps_out, compilation.canonical_target_irreps)
        if canonical_plan is not None
        else {}
    )
    active_deficit = coverage_deficit(
        compilation.active_plan.irreps_out,
        compilation.active_target_irreps,
    )
    if canonical_plan is not None and canonical_deficit:
        raise RuntimeError(
            "a reachable canonical plan contains an irrep coverage deficit"
        )
    if active_deficit:
        raise RuntimeError(
            "a completed compilation contains an active coverage deficit"
        )

    certificates = [
        CompilationCertificate(
            code="active_target_reachable",
            status="success",
            message="Every irrep and multiplicity required by the selected family is covered.",
            details={
                "active_depth": compilation.active_plan.depth,
                "active_deficit": {},
            },
        ).as_dict()
    ]
    if compilation.canonical_reachability.reachable:
        certificates.append(
            CompilationCertificate(
                code="canonical_reference_reachable",
                status="success",
                message="The unrestricted full-family reference is also reachable.",
                details={"canonical_depth": canonical_plan.depth},
            ).as_dict()
        )
    else:
        assert compilation.canonical_reachability.failure is not None
        failure = compilation.canonical_reachability.failure
        certificates.append(
            CompilationCertificate(
                code="canonical_reference_unreachable",
                status="diagnostic",
                message=(
                    "The unrestricted full-family reference is unreachable, but it is "
                    "diagnostic because the selected active parameter representation "
                    "is reachable."
                ),
                details={"underlying_failure": failure.as_dict()},
            ).as_dict()
        )
    covariance_is_full = (
        compilation.operator_family.relation_to_full.value == "equal_to_full"
    )
    if not covariance_is_full:
        certificates.append(
            CompilationCertificate(
                code="structured_covariance_restriction",
                status="restriction",
                message=(
                    f"The active {compilation.covariance_mode} family is a restriction "
                    "of the unrestricted operator family."
                ),
                details={
                    "selected_by": "operator_family_policy",
                    "budget": compilation.config.parameter_budget,
                    "canonical_coordinates": (
                        compilation.output_spec.dim
                        * (compilation.output_spec.dim + 1)
                        // 2
                    ),
                    "active_coordinates": compilation.covariance_parameter_count,
                },
            ).as_dict()
        )
    if compilation.backend_exact:
        certificates.append(
            CompilationCertificate(
                code="exact_execution_lowering",
                status="success",
                message="The execution backend preserves the compiled function space and checkpoint coordinates.",
                details={"backend": compilation.backend},
            ).as_dict()
        )
    else:
        certificates.append(
            CompilationCertificate(
                code="truncated_contraction",
                status="approximation",
                message="The requested contraction rank is not algebraically equivalent to the full CG executor.",
                details={
                    "backend": compilation.backend,
                    "contraction_rank": compilation.stf_contraction_rank,
                },
            ).as_dict()
        )

    parameter_counts = {
        "canonical_covariance_coordinates": (
            compilation.output_spec.dim * (compilation.output_spec.dim + 1) // 2
        ),
        "active_covariance_coordinates": compilation.covariance_parameter_count,
        "canonical_target_dimension": compilation.canonical_target_irreps.dim,
        "active_target_dimension": compilation.active_target_irreps.dim,
        "readout_trainable": _readout_parameter_count(compilation),
    }
    if executable is not None:
        parameter_counts["executable_trainable"] = sum(
            parameter.numel() for parameter in executable.parameters()
        )

    family_kind = {
        "full": "full_covariance",
        "block": "isotypic_block_covariance",
        "low_rank": "low_rank_plus_isotropic",
        "graph": "graph_precision",
    }.get(compilation.covariance_mode, compilation.covariance_mode)
    lowering_approximation = (
        None
        if compilation.backend_exact
        else {
            "kind": "truncated_multiplicity_rank",
            "requested_rank": compilation.stf_contraction_rank,
        }
    )
    output_record = _irrep_record(compilation.mean_irreps)
    output_record["cartesian_formula"] = compilation.output_spec.cartesian_formula
    seed_record = _irrep_record(compilation.seed_irreps)
    canonical_record = _irrep_record(compilation.canonical_target_irreps)
    active_record = _irrep_record(compilation.active_target_irreps)
    covariance_complexity = _covariance_complexity(compilation)
    probability = _probability_semantics(compilation)
    canonical_reachability = compilation.canonical_reachability.as_dict()
    active_reachability = compilation.active_reachability.as_dict()
    relation = compilation.operator_family.relation_to_full.value
    canonical_is_active = (
        compilation.canonical_target_irreps == compilation.active_target_irreps
    )
    family_record = {
        **compilation.operator_family.as_dict(),
        "kind": family_kind,
        "relation_to_canonical": (
            "canonical_full" if covariance_is_full else "strict_subfamily"
        ),
        "relation_to_full": relation,
    }
    optimization = match_optimized_program(compilation.operator_family)
    family_record["optimization"] = (
        optimization.as_dict()
        if optimization is not None
        else {
            "optimization_name": None,
            "fallback": "generic_recursive_interpreter",
            "reason": "no_exact_registered_template_match",
        }
    )
    fidelity_record = {
        "selected_executor": compilation.backend,
        "exactness": (
            "exact_for_active_family"
            if compilation.backend_exact
            else "approximate_for_active_family"
        ),
        "checkpoint_mapping": (
            "bijective" if compilation.backend_exact else "not_available"
        ),
        "approximation": lowering_approximation,
        "decision": compilation.executor_decision.capability.fidelity.as_dict(),
    }
    backend_selection_record = {
        "selected_executor": compilation.backend,
        **compilation.executor_decision.selection_basis,
        "capability_certificate": (compilation.executor_decision.capability.as_dict()),
    }
    record = {
        "schema_version": "3.0",
        "group": "O(3)",
        "output": output_record,
        "covariance_representation": _irrep_record(compilation.covariance_irreps),
        "seed": seed_record,
        "targets": {
            "canonical": canonical_record,
            "active": active_record,
        },
        "representation_reachability": {
            "status": "active_reachable",
            "proof_kind": "breadth_first_shortest_cg_paths",
            "canonical": canonical_reachability,
            "active": active_reachability,
            "multiplicity_coverage": {
                "canonical_deficit": (
                    {} if canonical_plan is not None else "not_computed_unreachable"
                ),
                "active_deficit": {},
            },
            "active_is_compilation_gate": True,
            "canonical_is_compilation_gate": canonical_is_active,
            "canonical_is_diagnostic": not canonical_is_active,
            "canonical_is_diagnostic_for_restricted_families": not covariance_is_full,
        },
        "execution_fidelity": fidelity_record,
        "backend_selection_basis": backend_selection_record,
        "family": family_record,
        "objective": probability["likelihood"],
        "complexity": {
            "covariance": covariance_complexity,
            "lifting_edges": compilation.active_plan.depth,
            "canonical_lifting_edges": (
                canonical_plan.depth if canonical_plan is not None else None
            ),
            "parameter_counts": parameter_counts,
        },
        "compatibility_hash": None,
        "probability": probability,
        "output_scope": compilation.config.output_scope,
        "certificates": certificates,
    }
    return CompilationReport.from_dict(record)

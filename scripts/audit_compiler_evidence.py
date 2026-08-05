"""Produce a compact, no-training audit of compiler-level evidence.

The audit reports compiler-plan quantities rather than model quality: output
dimension, reference/active parameter coordinates, target multiplicities,
retained CG type instructions, operator-IR nodes, planning time, and the
registered family relation.  It deliberately leaves application metrics and
GPU timing to their dedicated scripts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

from equivcompiler import (
    FeatureSpec,
    FullCovariance,
    GraphPrecision,
    IsotypicBlockCovariance,
    LowRankCovariance,
    plan_readout,
)
from representations import EquivariantOutputGraph
from representations.adaptive_lifting import irrep_multiplicities


def _count_ir_nodes(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    return int("kind" in node) + sum(
        _count_ir_nodes(child) for child in node.get("inputs", ())
    )


def _family_cases() -> tuple[tuple[str, str, object], ...]:
    graph = EquivariantOutputGraph(
        num_nodes=15,
        edges=tuple((index, index + 1) for index in range(14)),
        node_irrep="1o",
    )
    return (
        ("vector/full", "1o", FullCovariance()),
        ("rank2/full", "ij=ji", FullCovariance()),
        ("rank2/low_rank", "ij=ji", LowRankCovariance(2)),
        ("rank2/isotypic", "ij=ji", IsotypicBlockCovariance()),
        ("ITOP/graph", graph.output_irreps, GraphPrecision(graph)),
        ("elasticity/full", "ijkl=jikl=ijlk=klij", FullCovariance()),
        ("elasticity/low_rank", "ijkl=jikl=ijlk=klij", LowRankCovariance(8)),
    )


def _plan_record(name: str, plan, elapsed_ms: float) -> dict[str, Any]:
    compilation = plan.compilation
    report = plan.report.as_dict()
    family = report["family"]
    complexity = report["complexity"]
    counts = complexity["parameter_counts"]
    reference = complexity["canonical_cg_instructions"]
    active = complexity["retained_cg_instructions"]
    active_irreps = compilation.active_target_irreps
    multiplicities = irrep_multiplicities(active_irreps)
    return {
        "case": name,
        "output_irreps": str(compilation.mean_irreps),
        "output_dimension": compilation.output_spec.dim,
        "mean_target_dimension": compilation.mean_irreps.dim,
        "reference_operator_coordinates": counts["canonical_covariance_coordinates"],
        "active_operator_coordinates": counts["active_covariance_coordinates"],
        "reference_total_target_dimension": counts["canonical_target_dimension"],
        "active_total_target_dimension": counts["active_target_dimension"],
        # Keep the legacy names in the JSON schema for downstream readers.
        "active_target_dimension": counts["active_target_dimension"],
        "active_irrep_types": len(multiplicities),
        "active_irrep_multiplicity_total": sum(multiplicities.values()),
        "reference_parameter_coordinates": counts["canonical_covariance_coordinates"],
        "active_parameter_coordinates": counts["active_covariance_coordinates"],
        "reference_cg_depth": complexity["canonical_lifting_edges"],
        "active_cg_depth": complexity["lifting_edges"],
        "reference_cg_instructions": reference,
        "active_cg_instructions": active,
        "active_reference_instruction_ratio": (
            active / reference if reference not in (None, 0) else None
        ),
        "operator_ir_nodes": _count_ir_nodes(family["assembly_ir"]),
        "family": family["kind"],
        "family_relation": family["relation_to_full"],
        "domain": family["domain"],
        "planning_ms": elapsed_ms,
        "exact_executor": compilation.backend,
    }


def audit(seed_multiplicity: int, repeats: int) -> dict[str, Any]:
    seed = FeatureSpec.from_irreps(
        f"{2 * seed_multiplicity}x0e + {seed_multiplicity}x1o + "
        f"{seed_multiplicity}x2e",
        scope="global",
    )
    rows = []
    for name, output, family in _family_cases():
        measurements = []
        plan = None
        for _ in range(repeats):
            started = time.perf_counter()
            plan = plan_readout(seed, output=output, covariance=family)
            measurements.append((time.perf_counter() - started) * 1000.0)
        assert plan is not None
        rows.append(_plan_record(name, plan, median(measurements)))
    return {
        "kind": "compiler_evidence_audit",
        "training_steps": 0,
        "seed": {
            "irreps": str(seed.irreps),
            "multiplicity": seed_multiplicity,
            "scope": seed.scope,
        },
        "definitions": {
            "mean_target": "V, the equivariant predictive mean coordinates",
            "reference_operator": "Sym^2(V), the canonical full symmetric-operator coordinates",
            "active_operator": "P_F(V,S), the coordinates emitted by the selected operator family",
            "reference_total_target": "V + Sym^2(V), used for diagnostic reachability",
            "active_total_target": "V + P_F(V,S), the compiled trainable target",
            "active_failure_rule": "only active-target unreachability rejects compilation",
            "cg_instruction_unit": "retained irrep-type instruction, not FLOPs",
            "planning_time": "median plan_readout wall time; no training or GPU timing",
        },
        "repeats": repeats,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-multiplicity", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed_multiplicity < 1:
        parser.error("--seed-multiplicity must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    result = audit(args.seed_multiplicity, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

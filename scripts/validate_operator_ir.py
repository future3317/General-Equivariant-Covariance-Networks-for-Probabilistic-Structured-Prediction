"""No-training validation of the unified operator IR and executor autotuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from equivcompiler import (
    AutoBudget,
    BenchmarkTask,
    DeviceAutotuner,
    EllipticalDistribution,
    ExactExecutorCandidates,
    ExactOnly,
    FirstFeasible,
    FullCovariance,
    GraphPrecision,
    IsotypicBlockCovariance,
    LowRankCovariance,
    MinimizeLatency,
    PreferExecutor,
    SpecificExecutor,
    FeatureSpec,
    execution_signature_for_plan,
    plan_readout,
)
from representations import (
    EquivariantOutputGraph,
    O3IrrepsSpec,
    O3ProgramCompiler,
    analyze_lifting_graph,
)
from representations.operator_lowering import RecursiveOperatorMap
from scripts.benchmarking import environment_record


def _family_records() -> list[dict]:
    rank2 = O3IrrepsSpec.from_cartesian("ij=ji")
    graph = EquivariantOutputGraph(
        num_nodes=4,
        edges=((0, 1), (1, 2), (2, 3)),
        node_irrep="1o",
    )
    specifications = (
        ("full", FullCovariance(), rank2),
        ("low_rank", LowRankCovariance(2), rank2),
        ("isotypic_block", IsotypicBlockCovariance(), rank2),
        ("graph_precision", GraphPrecision(graph), O3IrrepsSpec(graph.output_irreps)),
    )
    return [
        {"name": name, **family.compile(output).as_dict()}
        for name, family, output in specifications
    ]


def _reachability_diagnostic() -> dict:
    output = O3IrrepsSpec.from_cartesian("ij=ji")
    family = IsotypicBlockCovariance().compile(output)
    compiler = O3ProgramCompiler(output)
    distribution = EllipticalDistribution()
    canonical = distribution.canonical_reference(output).decompose_o3().irreps
    active = distribution.active_parameter_rep(output, family).decompose_o3().irreps
    feature = FeatureSpec.from_irreps(output.irreps, scope="global")
    canonical_analysis = analyze_lifting_graph(feature.irreps, canonical, max_depth=0)
    active_analysis = analyze_lifting_graph(feature.irreps, active, max_depth=0)
    planned = plan_readout(
        feature,
        output=output,
        covariance=IsotypicBlockCovariance(),
        distribution=distribution,
    )
    compilation = compiler.compile(
        feature,
        operator_family=family,
        executor_decision=planned.compilation.executor_decision,
        distribution_spec=distribution,
        canonical_reachability=canonical_analysis,
        active_reachability=active_analysis,
    )
    report = compilation.report().as_dict()
    return {
        "search_contract": "zero tensor-product edges",
        "canonical": report["representation_reachability"]["canonical"],
        "active": report["representation_reachability"]["active"],
        "compilation_succeeded": True,
    }


def _selection_records(seed: FeatureSpec) -> dict:
    candidates = (
        FullCovariance(),
        LowRankCovariance(1),
        IsotypicBlockCovariance(),
    )
    automatic = plan_readout(
        seed,
        output="ij=ji",
        covariance=AutoBudget(30, candidates),
    )
    prioritized = plan_readout(
        seed,
        output="ij=ji",
        covariance=FirstFeasible(30, candidates),
    )
    return {
        "auto_budget": automatic.report.family,
        "first_feasible": prioritized.report.family,
    }


def _recursive_oracle_validation(
    seed: FeatureSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    graph = EquivariantOutputGraph(
        num_nodes=4,
        edges=((0, 1), (1, 2), (2, 3)),
        node_irrep="1o",
    )
    cases = (
        ("full", "ij=ji", FullCovariance()),
        ("low_rank", "ij=ji", LowRankCovariance(2)),
        ("isotypic_block", "ij=ji", IsotypicBlockCovariance()),
        ("graph_precision", graph.output_irreps, GraphPrecision(graph)),
    )
    records = []
    for name, output, family in cases:
        plan = plan_readout(seed, output=output, covariance=family)
        optimized = plan.compilation.build_spd_map().to(device=device, dtype=dtype)
        recursive = RecursiveOperatorMap(plan.compilation).to(
            device=device, dtype=dtype
        )
        left = (
            0.1
            * torch.randn(
                3,
                plan.compilation.covariance_parameter_count,
                device=device,
                dtype=dtype,
            )
        ).requires_grad_()
        right = left.detach().clone().requires_grad_()
        residual = torch.randn(
            3,
            plan.compilation.output_spec.dim,
            device=device,
            dtype=dtype,
        )
        optimized_covariance = optimized(left)
        recursive_covariance = recursive(right)
        optimized_stats = optimized.statistics(left, residual)
        recursive_stats = recursive.statistics(right, residual)
        optimized_loss = optimized_covariance.square().mean() + sum(
            value.mean() for value in optimized_stats
        )
        recursive_loss = recursive_covariance.square().mean() + sum(
            value.mean() for value in recursive_stats
        )
        optimized_gradient = torch.autograd.grad(optimized_loss, left)[0]
        recursive_gradient = torch.autograd.grad(recursive_loss, right)[0]
        records.append(
            {
                "family": name,
                "optimization": optimized.optimization_name,
                "covariance_max_abs": float(
                    (optimized_covariance - recursive_covariance).abs().max().detach()
                ),
                "logdet_max_abs": float(
                    (optimized_stats[0] - recursive_stats[0]).abs().max().detach()
                ),
                "quadratic_max_abs": float(
                    (optimized_stats[1] - recursive_stats[1]).abs().max().detach()
                ),
                "parameter_gradient_max_abs": float(
                    (optimized_gradient - recursive_gradient).abs().max().detach()
                ),
            }
        )
    return records


def _build_exact_pair(seed: FeatureSpec, device: torch.device, dtype: torch.dtype):
    common = dict(
        output="ij=ji",
        covariance=FullCovariance(),
        fidelity=ExactOnly(),
        output_scope="global",
    )
    spherical_plan = plan_readout(
        seed,
        executor=SpecificExecutor("spherical_cg"),
        cost=PreferExecutor(("spherical_cg",)),
        **common,
    )
    cartesian_plan = plan_readout(
        seed,
        executor=SpecificExecutor("cartesian_stf"),
        cost=PreferExecutor(("cartesian_stf",)),
        **common,
    )
    spherical = spherical_plan.build_readout(device=device, dtype=dtype)
    cartesian = cartesian_plan.build_readout(device=device, dtype=dtype)
    cartesian.load_state_dict(spherical.state_dict(), strict=True)
    return spherical_plan, spherical, cartesian


def _executor_validation(
    seed: FeatureSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict:
    spherical_plan, spherical, cartesian = _build_exact_pair(seed, device, dtype)
    features = (0.1 * seed.irreps.randn(batch_size, -1, device=device, dtype=dtype)).requires_grad_()
    with torch.no_grad():
        spherical_output = spherical(features, return_scale=True)
        cartesian_output = cartesian(features, return_scale=True)
    maximum_error = {
        key: float((spherical_output[key] - cartesian_output[key]).abs().max())
        for key in ("mu", "params", "scale")
    }
    relative_error = {
        key: float(
            (spherical_output[key] - cartesian_output[key]).abs().max()
            / spherical_output[key].abs().max().clamp_min(
                torch.finfo(dtype).tiny
            )
        )
        for key in ("mu", "params", "scale")
    }

    phase = "forward_backward"
    signature = execution_signature_for_plan(
        spherical_plan,
        batch_shape=(batch_size, seed.irreps.dim),
        dtype=str(dtype).removeprefix("torch."),
        device=device,
        phase=phase,
    )
    latest = None

    def task(module):
        def prepare():
            module.zero_grad(set_to_none=True)
            features.grad = None

        def run_once():
            nonlocal latest
            latest = module(features)
            (latest["mu"].square().mean() + latest["params"].square().mean()).backward()
            return latest

        return BenchmarkTask(run_once, prepare)

    measurements = DeviceAutotuner(warmup, repeats).benchmark(
        signature,
        {
            "spherical_cg": task(spherical),
            "cartesian_stf": task(cartesian),
        },
    )
    selected = plan_readout(
        seed,
        output="ij=ji",
        covariance=FullCovariance(),
        fidelity=ExactOnly(),
        executor=ExactExecutorCandidates(),
        cost=MinimizeLatency(signature, measurements),
    )
    return {
        "signature": signature.as_dict(),
        "maximum_absolute_error": maximum_error,
        "maximum_relative_error": relative_error,
        "measurements": [
            {
                "executor": item.executor,
                "median_ms": item.median_ms,
                "iqr_ms": item.iqr_ms,
            }
            for item in measurements
        ],
        "selected_executor": selected.compilation.backend,
        "selection_basis": selected.report.backend_selection_basis,
        "strict_checkpoint_load_succeeded": True,
        "learned_parameter_keys_identical": (
            set(dict(spherical.named_parameters()))
            == set(dict(cartesian.named_parameters()))
        ),
    }


def validate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = {
        "float64": torch.float64,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    seed = FeatureSpec.from_irreps(
        f"{2 * args.multiplicity}x0e + {args.multiplicity}x1o + {args.multiplicity}x2e",
        scope="global",
    )
    return {
        "kind": "operator_ir_and_executor_validation",
        "training_steps": 0,
        "environment": environment_record(device),
        "representation_pipeline": ["RepExpr", "DecomposedRep", "ConcreteLayout"],
        "operator_families": _family_records(),
        "family_selection": _selection_records(seed),
        "reachability_gate": _reachability_diagnostic(),
        "recursive_oracle_equivalence": _recursive_oracle_validation(
            seed,
            device=device,
            dtype=dtype,
        ),
        "executor_autotune": _executor_validation(
            seed,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype", choices=("float64", "float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--multiplicity", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(args)
    rendered = json.dumps(result, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

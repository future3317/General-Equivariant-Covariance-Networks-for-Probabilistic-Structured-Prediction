"""Target-directed Clebsch--Gordan lifting for O(3) representations.

The planner searches irrep *types* rather than imposing an angular-momentum
cutoff. Consequently angular momentum, parity and target multiplicity are all
part of the compiled representation frontier. Every retained intermediate
type lies on a shortest CG path to a requested output irrep. Execution remains
fully connected over admissible CG instructions between retained frontiers;
the plan is therefore depth-minimal and target-pruned, not globally minimal in
frontier width, instruction count, FLOPs, or parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from compatibility.e3nn import o3

from representations.dense_projector import MultiplicityFirstDenseTensorProduct
from representations.diagnostics import (
    CompilationCertificate,
    CompilationError,
    UnreachableTargetError,
)


def irrep_multiplicities(irreps: o3.Irreps) -> dict[o3.Irrep, int]:
    """Return total multiplicity for every irrep type."""
    counts: dict[o3.Irrep, int] = {}
    for multiplicity, irrep in o3.Irreps(irreps):
        counts[irrep] = counts.get(irrep, 0) + multiplicity
    return counts


def irreps_from_multiplicities(
    counts: dict[o3.Irrep, int],
) -> o3.Irreps:
    """Create a deterministic, simplified irrep layout from multiplicities."""
    terms = [
        (multiplicity, irrep)
        for irrep, multiplicity in sorted(
            counts.items(), key=lambda item: (item[0].l, -item[0].p)
        )
        if multiplicity > 0
    ]
    return o3.Irreps(terms)


def direct_sum_irreps(*representations: o3.Irreps) -> o3.Irreps:
    """Canonical direct sum with equal irrep types merged."""
    counts: dict[o3.Irrep, int] = {}
    for representation in representations:
        for irrep, multiplicity in irrep_multiplicities(representation).items():
            counts[irrep] = counts.get(irrep, 0) + multiplicity
    return irreps_from_multiplicities(counts)


def coverage_deficit(
    source_irreps: o3.Irreps, target_irreps: o3.Irreps
) -> dict[o3.Irrep, int]:
    """Return ``target - source`` multiplicity deficits by irrep type."""
    source = irrep_multiplicities(source_irreps)
    target = irrep_multiplicities(target_irreps)
    return {
        irrep: multiplicity - source.get(irrep, 0)
        for irrep, multiplicity in target.items()
        if multiplicity > source.get(irrep, 0)
    }


def _cg_outputs(left: o3.Irrep, right: o3.Irrep, max_l: int) -> Iterable[o3.Irrep]:
    parity = left.p * right.p
    upper = min(left.l + right.l, max_l)
    for angular_momentum in range(abs(left.l - right.l), upper + 1):
        yield o3.Irrep(angular_momentum, parity)


@dataclass(frozen=True)
class LiftingStage:
    """One edge ``H^(depth-1) -> H^depth`` in a lifting graph."""

    depth: int
    irreps_in: o3.Irreps
    irreps_out: o3.Irreps


@dataclass(frozen=True)
class O3LiftingPlan:
    """A depth-minimal CG plan pruned to its selected shortest paths."""

    seed_irreps: o3.Irreps
    target_irreps: o3.Irreps
    stages: tuple[LiftingStage, ...]
    paths: dict[str, tuple[str, ...]]

    @property
    def depth(self) -> int:
        return len(self.stages)

    @property
    def irreps_out(self) -> o3.Irreps:
        return self.stages[-1].irreps_out if self.stages else self.target_irreps

    def as_dict(self) -> dict:
        return {
            "seed_irreps": str(self.seed_irreps),
            "target_irreps": str(self.target_irreps),
            "depth": self.depth,
            "minimality": "tensor_product_depth",
            "pruning": "union_of_selected_shortest_path_frontiers",
            "execution": "fully_connected_within_retained_frontiers",
            "stages": [
                {
                    "depth": stage.depth,
                    "irreps_in": str(stage.irreps_in),
                    "irreps_out": str(stage.irreps_out),
                }
                for stage in self.stages
            ],
            "paths": self.paths,
        }


@dataclass(frozen=True)
class O3ReachabilityAnalysis:
    """Non-throwing reachability result for canonical or active diagnostics."""

    seed_irreps: o3.Irreps
    target_irreps: o3.Irreps
    plan: O3LiftingPlan | None
    failure: CompilationCertificate | None = None

    @property
    def reachable(self) -> bool:
        return self.plan is not None

    def as_dict(self) -> dict:
        return {
            "reachable": self.reachable,
            "seed_irreps": str(self.seed_irreps),
            "target_irreps": str(self.target_irreps),
            "depth": self.plan.depth if self.plan is not None else None,
            "missing_irreps": (
                []
                if self.failure is None
                else list(self.failure.details.get("missing_irreps", []))
            ),
            "failure": self.failure.as_dict() if self.failure is not None else None,
            "plan": self.plan.as_dict() if self.plan is not None else None,
        }


def plan_lifting_graph(
    seed_irreps: o3.Irreps,
    target_irreps: o3.Irreps,
    *,
    max_depth: int | None = None,
) -> O3LiftingPlan:
    """Compile a shortest target-directed CG graph.

    Search is breadth first, hence the returned tensor-product depth is
    minimal. Ties at the same depth prefer lower-dimensional CG paths. The
    representation frontier is pruned to the union of selected shortest paths;
    target multiplicities are carried explicitly through every retained path.
    ``_O3LiftingStage`` deliberately uses a fully connected tensor product over
    these retained types, so this routine does not claim globally minimal
    frontier width, instruction count, FLOPs, or parameterization.

    Raises:
        ValueError: if a target parity/angular momentum cannot be generated
            from the seed representation.
    """
    seed = o3.Irreps(seed_irreps)
    target = o3.Irreps(target_irreps)
    seed_counts = irrep_multiplicities(seed)
    target_counts = irrep_multiplicities(target)
    if not seed_counts:
        raise CompilationError(
            CompilationCertificate(
                code="empty_seed_representation",
                status="failure",
                message="seed_irreps must not be empty",
                details={"seed_irreps": str(seed), "target_irreps": str(target)},
            )
        )
    if not target_counts:
        raise CompilationError(
            CompilationCertificate(
                code="empty_target_representation",
                status="failure",
                message="target_irreps must not be empty",
                details={"seed_irreps": str(seed), "target_irreps": str(target)},
            )
        )

    target_lmax = max(irrep.l for irrep in target_counts)
    search_lmax = max(target_lmax, max(irrep.l for irrep in seed_counts))
    if max_depth is None:
        max_depth = max(2, 2 * search_lmax + 2)

    distance = {irrep: 0 for irrep in seed_counts}
    cost = {irrep: 0 for irrep in seed_counts}
    parent: dict[o3.Irrep, tuple[o3.Irrep, o3.Irrep]] = {}
    frontier = set(seed_counts)

    for depth in range(1, max_depth + 1):
        candidates: dict[o3.Irrep, tuple[int, o3.Irrep, o3.Irrep]] = {}
        for current in frontier:
            for factor in seed_counts:
                edge_cost = (2 * current.l + 1) * (2 * factor.l + 1)
                for output in _cg_outputs(current, factor, search_lmax):
                    candidate_cost = cost[current] + edge_cost * (2 * output.l + 1)
                    previous = candidates.get(output)
                    if previous is None or candidate_cost < previous[0]:
                        candidates[output] = (candidate_cost, current, factor)

        next_frontier: set[o3.Irrep] = set()
        for output, (candidate_cost, current, factor) in candidates.items():
            if output not in distance:
                distance[output] = depth
                cost[output] = candidate_cost
                parent[output] = (current, factor)
                next_frontier.add(output)

        if all(irrep in distance for irrep in target_counts):
            break
        if not next_frontier:
            break
        frontier = next_frontier

    unreachable = [irrep for irrep in target_counts if irrep not in distance]
    if unreachable:
        missing = ", ".join(str(irrep) for irrep in unreachable)
        seed_parities = {irrep.p for irrep in seed_counts}
        missing_parities = {irrep.p for irrep in unreachable}
        parity_obstruction = seed_parities == {1} and -1 in missing_parities
        angular_obstruction = max(irrep.l for irrep in seed_counts) == 0 and any(
            irrep.l > 0 for irrep in unreachable
        )
        if parity_obstruction:
            code = "parity_unreachable"
            reason = "available parity paths cannot produce the missing odd irreps"
            remedies = [
                "expose at least one odd-parity seed channel",
                "explicitly select a restricted covariance family if scientifically justified",
            ]
        elif angular_obstruction:
            code = "angular_momentum_unreachable"
            reason = "scalar seed factors cannot generate non-scalar angular momentum"
            remedies = [
                "expose a non-scalar seed channel",
                "increase the backbone angular cutoff",
                "explicitly select a restricted covariance family if scientifically justified",
            ]
        else:
            code = "target_unreachable"
            reason = "no CG path was found within the declared search contract"
            remedies = [
                "increase the backbone angular cutoff or expose additional parity types",
                "inspect the missing_irrep_multiplicities field",
                "explicitly select a restricted covariance family if scientifically justified",
            ]
        raise UnreachableTargetError(
            CompilationCertificate(
                code=code,
                status="failure",
                message=(
                    f"target irreps [{missing}] are unreachable from seed {seed}; "
                    "the seed lacks the required angular-momentum/parity CG paths"
                ),
                details={
                    "seed_irreps": str(seed),
                    "target_irreps": str(target),
                    "missing_irreps": [str(irrep) for irrep in unreachable],
                    "missing_irrep_multiplicities": {
                        str(irrep): target_counts[irrep] for irrep in unreachable
                    },
                    "max_depth": max_depth,
                    "parity_obstruction": parity_obstruction,
                    "angular_momentum_obstruction": angular_obstruction,
                    "reason": reason,
                    "possible_remedies": remedies,
                    "restricted_family_changes_statistical_model": True,
                },
            )
        )

    typed_paths: dict[o3.Irrep, list[o3.Irrep]] = {}
    for target_irrep in target_counts:
        path = [target_irrep]
        cursor = target_irrep
        while distance[cursor] > 0:
            cursor = parent[cursor][0]
            path.append(cursor)
        typed_paths[target_irrep] = list(reversed(path))

    depth = max(distance[irrep] for irrep in target_counts)
    stages: list[LiftingStage] = []
    current_irreps = seed
    for stage_depth in range(1, depth + 1):
        stage_counts: dict[o3.Irrep, int] = {}
        for target_irrep, multiplicity in target_counts.items():
            path = typed_paths[target_irrep]
            state = path[min(stage_depth, len(path) - 1)]
            stage_counts[state] = max(stage_counts.get(state, 0), multiplicity)
        next_irreps = (
            target if stage_depth == depth else irreps_from_multiplicities(stage_counts)
        )
        stages.append(LiftingStage(stage_depth, current_irreps, next_irreps))
        current_irreps = next_irreps

    paths = {
        str(irrep): tuple(str(state) for state in path)
        for irrep, path in typed_paths.items()
    }
    return O3LiftingPlan(seed, target, tuple(stages), paths)


def required_lifting_depth(seed_irreps: o3.Irreps, target_irreps: o3.Irreps) -> int:
    """Return the exact number of required tensor-product edges."""
    return plan_lifting_graph(seed_irreps, target_irreps).depth


def analyze_lifting_graph(
    seed_irreps: o3.Irreps,
    target_irreps: o3.Irreps,
    *,
    max_depth: int | None = None,
) -> O3ReachabilityAnalysis:
    """Analyze reachability without using failure as a family-selection event."""
    seed = o3.Irreps(seed_irreps)
    target = o3.Irreps(target_irreps)
    try:
        plan = plan_lifting_graph(seed, target, max_depth=max_depth)
    except UnreachableTargetError as error:
        return O3ReachabilityAnalysis(seed, target, None, error.certificate)
    return O3ReachabilityAnalysis(seed, target, plan)


class O3AdaptiveLifting(torch.nn.Module):
    """Execute a compiled target-directed CG graph."""

    def __init__(
        self,
        seed_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        *,
        plan: O3LiftingPlan | None = None,
        tensor_product_backend: str = "spherical_cg",
        contraction_rank: int | None = None,
    ):
        super().__init__()
        if tensor_product_backend not in {"spherical_cg", "dense_projector"}:
            raise ValueError(
                "tensor_product_backend must be spherical_cg or dense_projector"
            )
        self.plan = plan or plan_lifting_graph(seed_irreps, target_irreps)
        if o3.Irreps(seed_irreps) != self.plan.seed_irreps:
            raise ValueError("seed_irreps does not match the supplied lifting plan")
        if o3.Irreps(target_irreps) != self.plan.target_irreps:
            raise ValueError("target_irreps does not match the supplied lifting plan")

        self.seed_irreps = self.plan.seed_irreps
        self.target_irreps = self.plan.target_irreps
        self.depth = self.plan.depth
        self.stages = torch.nn.ModuleList()
        for stage in self.plan.stages:
            self.stages.append(
                _O3LiftingStage(
                    stage.irreps_in,
                    self.seed_irreps,
                    stage.irreps_out,
                    tensor_product_backend=tensor_product_backend,
                    contraction_rank=contraction_rank,
                )
            )
        self.final_linear = (
            o3.Linear(self.seed_irreps, self.target_irreps)
            if self.depth == 0
            else torch.nn.Identity()
        )
        self.irreps_out = self.target_irreps

    def forward(self, seed_features: torch.Tensor) -> torch.Tensor:
        if seed_features.shape[-1] != self.seed_irreps.dim:
            raise ValueError(
                f"expected seed dimension {self.seed_irreps.dim}, "
                f"got {seed_features.shape[-1]}"
            )
        if self.depth == 0:
            return self.final_linear(seed_features)
        hidden = seed_features
        for stage in self.stages:
            hidden = stage(hidden, seed_features)
        return hidden


class _O3LiftingStage(torch.nn.Module):
    """Residual linear transport plus one CG tensor-product edge."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        seed_irreps: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        tensor_product_backend: str = "spherical_cg",
        contraction_rank: int | None = None,
    ):
        super().__init__()
        self.linear = o3.Linear(irreps_in, irreps_out)
        if tensor_product_backend == "dense_projector":
            self.tensor_product = MultiplicityFirstDenseTensorProduct(
                irreps_in,
                seed_irreps,
                irreps_out,
                contraction_rank=contraction_rank,
            )
        else:
            self.tensor_product = o3.FullyConnectedTensorProduct(
                irreps_in,
                seed_irreps,
                irreps_out,
                internal_weights=True,
                shared_weights=True,
                irrep_normalization="component",
                path_normalization="element",
            )

    def forward(self, hidden: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden) + self.tensor_product(hidden, seed)

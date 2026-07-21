"""Registry and candidate enumeration for exact execution lowerings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from representations import O3IrrepsSpec
from representations.cartesian_stf import is_rank2_stf_output


@dataclass(frozen=True)
class ExecutionContext:
    """Typed facts used to decide whether an executor can lower a program."""

    output: O3IrrepsSpec
    operator_family: str
    active_depth: int


class ExactLoweringRegistry:
    """Small explicit registry; it does not encode a performance ordering."""

    def __init__(self) -> None:
        self._predicates: dict[str, Callable[[ExecutionContext], bool]] = {}

    def register(
        self, name: str, predicate: Callable[[ExecutionContext], bool]
    ) -> None:
        if name in self._predicates:
            raise ValueError(f"executor {name!r} is already registered")
        self._predicates[name] = predicate

    def supports(self, name: str, context: ExecutionContext) -> bool:
        try:
            predicate = self._predicates[name]
        except KeyError as error:
            raise ValueError(f"unknown executor {name!r}") from error
        return bool(predicate(context))

    def as_dict(self, context: ExecutionContext) -> dict[str, bool]:
        return {
            name: bool(predicate(context))
            for name, predicate in self._predicates.items()
        }


class CandidateEnumerator:
    """Filter requested executor names by registry support."""

    def __init__(self, registry: ExactLoweringRegistry) -> None:
        self.registry = registry

    def enumerate(
        self, requested: tuple[str, ...], context: ExecutionContext
    ) -> tuple[str, ...]:
        return tuple(
            name for name in requested if self.registry.supports(name, context)
        )


DEFAULT_EXACT_LOWERINGS = ExactLoweringRegistry()
DEFAULT_EXACT_LOWERINGS.register("spherical_cg", lambda context: True)
DEFAULT_EXACT_LOWERINGS.register(
    "cartesian_stf",
    lambda context: (
        context.operator_family == "full"
        and is_rank2_stf_output(context.output.irreps)
        and context.active_depth == 1
    ),
)

"""Named representation-decomposition backends.

The released backend is O(3)-specialized.  Keeping the dispatch boundary
separate from the expression classes makes that scope explicit and provides a
single registration point for a future backend without changing expression
semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from representations.representation_ir import DecomposedRep, RepExpr


class RepresentationBackend(Protocol):
    """Backend contract for decomposing a semantic representation expression."""

    name: str

    def decompose(self, expression: RepExpr) -> DecomposedRep:
        ...


class O3RepresentationBackend:
    """The currently released e3nn-compatible O(3) decomposition backend."""

    name = "o3"

    def decompose(self, expression: RepExpr) -> DecomposedRep:
        return expression.decompose_o3()


_BACKENDS: dict[str, RepresentationBackend] = {"o3": O3RepresentationBackend()}


def get_representation_backend(name: str) -> RepresentationBackend:
    try:
        return _BACKENDS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported representation backend: {name}") from exc

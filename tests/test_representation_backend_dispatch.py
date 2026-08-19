from __future__ import annotations

import pytest

from representations import IrrepsExpr


def test_representation_expression_dispatches_through_named_o3_backend():
    expression = IrrepsExpr("0e+2e")

    direct = expression.decompose_o3()
    dispatched = expression.decompose(backend="o3")

    assert dispatched == direct
    assert dispatched.group == "O3"


def test_representation_expression_rejects_unregistered_backend():
    with pytest.raises(ValueError, match="unsupported representation backend"):
        IrrepsExpr("0e").decompose(backend="so3")

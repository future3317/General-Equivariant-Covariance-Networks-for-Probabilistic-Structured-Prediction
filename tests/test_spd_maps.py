"""Tests for SPD maps."""

import pytest
import torch

from gecn.spd_maps import (
    MatrixExponentialMap,
    SpectralSoftplusMap,
    SquarePlusIdentityMap,
    PrecisionExponentialMap,
    LowRankPlusIsotropicMap,
)


MAP_CLASSES = [
    MatrixExponentialMap,
    SpectralSoftplusMap,
    SquarePlusIdentityMap,
    PrecisionExponentialMap,
]


def _symmetric_matrix(batch, d):
    A_raw = torch.randn(batch, d, d, requires_grad=True)
    A = 0.5 * (A_raw + A_raw.transpose(-1, -2))
    return A, A_raw


@pytest.mark.parametrize("MapClass", MAP_CLASSES)
def test_spd_output(MapClass):
    A, _ = _symmetric_matrix(4, 6)
    spdm = MapClass()
    S = spdm(A)
    eigs = torch.linalg.eigvalsh(S.detach())
    assert eigs.min().item() > 0


@pytest.mark.parametrize("MapClass", MAP_CLASSES)
def test_spd_gradients_finite(MapClass):
    A, A_raw = _symmetric_matrix(4, 6)
    r = torch.randn(4, 6)
    spdm = MapClass()
    S = spdm(A)
    ld = spdm.logdet(A)
    pa = spdm.precision_action(A, r)
    loss = S.sum() + ld.sum() + pa.sum()
    loss.backward()
    assert A_raw.grad is not None
    assert torch.isfinite(A_raw.grad).all()


@pytest.mark.parametrize("MapClass", MAP_CLASSES)
def test_degenerate_gradient_finite(MapClass):
    A = torch.eye(6).unsqueeze(0).expand(4, 6, 6).contiguous() * 2.0
    A.requires_grad_(True)
    spdm = MapClass()
    S = spdm(A)
    S.sum().backward()
    assert torch.isfinite(A.grad).all()


def test_low_rank_spd():
    params = torch.randn(4, 6 * 3 + 1, requires_grad=True)
    spdm = LowRankPlusIsotropicMap(dim=6, rank=3)
    S = spdm(params)
    eigs = torch.linalg.eigvalsh(S.detach())
    assert eigs.min().item() > 0
    S.sum().backward()
    assert torch.isfinite(params.grad).all()


def test_no_anisotropic_jitter_in_package():
    """Ensure the forbidden anisotropic eigenvalue jitter is not in gecn/."""
    import gecn
    import inspect
    import pathlib
    root = pathlib.Path(inspect.getfile(gecn)).parent
    for pyfile in root.rglob("*.py"):
        text = pyfile.read_text(encoding="utf-8")
        assert "anisotropic" not in text.lower(), f"anisotropic jitter found in {pyfile}"
        assert "safe_eigh" not in text, f"safe_eigh found in {pyfile}"

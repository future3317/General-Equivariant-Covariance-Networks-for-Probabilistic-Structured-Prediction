"""Tests for the synthetic covariance-recovery experiment."""

import pytest
import torch
from e3nn import o3

from experiments.synthetic_covariance_recovery import (
    DEFAULT_INPUT_IRREPS,
    EquivariantTeacher,
    SyntheticBackbone,
    SyntheticDataset,
    _make_data_object,
    evaluate,
)
from representations import O3IrrepsSpec
from representations.symmetric_square import O3SymmetricOperatorBasis
from spd_maps import MatrixExponentialMap
from distributions import GaussianNLL
from models import (
    EquivariantMeanHead,
    O3QuadraticSymmetricOperatorHead,
    StructuredProbabilisticPredictor,
)


def _get_l4_slice(operator_basis: O3SymmetricOperatorBasis):
    cursor = 0
    for mul, ir in operator_basis.operator_irreps:
        dim = ir.dim
        for _ in range(mul):
            if ir.l == 4:
                return slice(cursor, cursor + dim)
            cursor += dim
    return None


@pytest.mark.parametrize("output_irreps", ["1o", "0e + 2e"])
def test_synthetic_dataset_generates_spd(output_irreps):
    """Generated covariance matrices must be strictly SPD and finite."""
    output_spec = O3IrrepsSpec(output_irreps)
    teacher = EquivariantTeacher(DEFAULT_INPUT_IRREPS, output_spec)
    ds = SyntheticDataset(output_irreps, num_samples=32, teacher=teacher, seed=0)
    x, y, mu, A, S = ds.generate()

    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(A).all()
    assert torch.isfinite(S).all()

    eigs = torch.linalg.eigvalsh(S)
    assert eigs.min().item() > 0


@pytest.mark.parametrize("output_irreps", ["1o", "0e + 2e"])
def test_synthetic_model_forward_backward(output_irreps):
    """The full predictor can be built and trained for one step."""
    output_spec = O3IrrepsSpec(output_irreps)
    input_irreps = o3.Irreps("16x0e + 8x1o + 4x2e")
    backbone = SyntheticBackbone(input_irreps=input_irreps, hidden_irreps="16x0e + 8x1o + 4x2e")
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3QuadraticSymmetricOperatorHead(backbone.irreps_out, output_spec, pool=True)

    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=MatrixExponentialMap(),
        distribution=GaussianNLL(),
    )

    teacher = EquivariantTeacher(input_irreps, output_spec)
    ds = SyntheticDataset(output_irreps, num_samples=16, teacher=teacher, seed=1)
    x, y, mu_true, A_true, S_true = ds.generate()
    data = _make_data_object(x)

    result = model(data, target=y, return_scale=True)
    assert "loss" in result
    assert "mu" in result
    assert "scale" in result
    assert result["mu"].shape == (16, output_spec.dim)
    assert result["scale"].shape == (16, output_spec.dim, output_spec.dim)

    result["loss"].backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("output_irreps", ["1o", "0e + 2e"])
def test_evaluate_metrics_finite(output_irreps):
    """Evaluation metrics must all be finite."""
    output_spec = O3IrrepsSpec(output_irreps)
    teacher = EquivariantTeacher(DEFAULT_INPUT_IRREPS, output_spec)
    ds = SyntheticDataset(output_irreps, num_samples=16, teacher=teacher, seed=2)
    x, y, mu_true, A_true, S_true = ds.generate()

    # Use the ground truth as predictions for a sanity check.
    metrics = evaluate(mu_true, A_true, y, mu_true, A_true, S_true)
    for name, value in metrics.items():
        assert isinstance(value, float), name
        assert torch.isfinite(torch.tensor(value)), name


def test_synthetic_teacher_equivariance():
    """Teacher's A and S must transform equivariantly under O(3)."""
    output_spec = O3IrrepsSpec("0e + 2e")
    teacher = EquivariantTeacher(DEFAULT_INPUT_IRREPS, output_spec)

    x = teacher.input_irreps.randn(16, -1)
    R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    rho_in = teacher.input_irreps.D_from_matrix(R)
    x_rot = (rho_in @ x.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        mu, A, S = teacher(x)
        mu_rot, A_rot, S_rot = teacher(x_rot)

    rho_out = output_spec.representation_matrix(R)

    mu_err = torch.norm(mu_rot - mu @ rho_out.T, dim=-1).max().item()
    A_err = torch.norm(A_rot - rho_out @ A @ rho_out.T, dim=(-2, -1)).max().item()
    S_err = torch.norm(S_rot - rho_out @ S @ rho_out.T, dim=(-2, -1)).max().item()

    assert mu_err < 1e-5
    assert A_err < 1e-5
    assert S_err < 1e-5


def test_synthetic_shared_teacher():
    """Train and test datasets with different seeds must share the same teacher."""
    output_spec = O3IrrepsSpec("0e + 2e")
    teacher = EquivariantTeacher(DEFAULT_INPUT_IRREPS, output_spec)
    ds_train = SyntheticDataset("0e + 2e", num_samples=16, teacher=teacher, seed=0)
    ds_test = SyntheticDataset("0e + 2e", num_samples=16, teacher=teacher, seed=1)

    assert ds_train.teacher is ds_test.teacher

    x_train, y_train, mu_train, A_train, S_train = ds_train.generate()
    x_test, y_test, mu_test, A_test, S_test = ds_test.generate()

    # Different seeds -> different input samples.
    assert not torch.allclose(x_train, x_test)


def test_synthetic_teacher_has_nonzero_4e():
    """Quadratic teacher must produce nonzero 4e coefficients for rank-2 output."""
    output_spec = O3IrrepsSpec("0e + 2e")
    teacher = EquivariantTeacher(DEFAULT_INPUT_IRREPS, output_spec)

    x = teacher.input_irreps.randn(64, -1)
    with torch.no_grad():
        _, A, _ = teacher(x)

    operator_basis = output_spec.symmetric_square()
    coeffs = operator_basis.project(A)
    l4_slice = _get_l4_slice(operator_basis)
    assert l4_slice is not None
    assert coeffs[..., l4_slice].abs().max().item() > 1e-6


def test_synthetic_student_4e_gradient():
    """Student's 4e branch must receive nonzero finite gradients."""
    output_spec = O3IrrepsSpec("0e + 2e")
    input_irreps = o3.Irreps("16x0e + 8x1o + 4x2e")
    backbone = SyntheticBackbone(input_irreps=input_irreps, hidden_irreps="32x0e + 16x1o + 16x2e")
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3QuadraticSymmetricOperatorHead(backbone.irreps_out, output_spec, pool=True)
    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=MatrixExponentialMap(),
        distribution=GaussianNLL(),
    )

    teacher = EquivariantTeacher(input_irreps, output_spec)
    ds = SyntheticDataset("0e + 2e", num_samples=16, teacher=teacher, seed=3)
    x, y, _, _, _ = ds.generate()
    data = _make_data_object(x)

    result = model(data, target=y, return_scale=False)
    result["loss"].backward()

    # Accumulate gradient norm in the 4e slice of the quadratic square branch.
    operator_basis = output_spec.symmetric_square()
    l4_slice = _get_l4_slice(operator_basis)
    assert l4_slice is not None

    square_grad_norm = 0.0
    for p in cov_head.square.parameters():
        if p.grad is not None:
            square_grad_norm += p.grad.pow(2).sum().item()
    assert square_grad_norm > 1e-12


def test_synthetic_rank2_full_operator_recovery():
    """Training should reduce 4e coefficient error relative to an untrained model."""
    output_spec = O3IrrepsSpec("0e + 2e")
    input_irreps = o3.Irreps("16x0e + 8x1o + 4x2e")
    teacher = EquivariantTeacher(input_irreps, output_spec)

    def _build_model():
        backbone = SyntheticBackbone(input_irreps=input_irreps, hidden_irreps="32x0e + 16x1o + 16x2e")
        mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
        cov_head = O3QuadraticSymmetricOperatorHead(backbone.irreps_out, output_spec, pool=True)
        return StructuredProbabilisticPredictor(
            backbone=backbone,
            output_spec=output_spec,
            mean_head=mean_head,
            covariance_head=cov_head,
            spd_map=MatrixExponentialMap(),
            distribution=GaussianNLL(),
        )

    def _l4_error(model, x, true_A):
        with torch.no_grad():
            pred_A = model(_make_data_object(x), return_scale=False)["params"]
        operator_basis = output_spec.symmetric_square()
        true_coeffs = operator_basis.project(true_A)
        pred_coeffs = operator_basis.project(pred_A)
        l4_slice = _get_l4_slice(operator_basis)
        return (
            torch.norm(pred_coeffs[..., l4_slice] - true_coeffs[..., l4_slice])
            / (torch.norm(true_coeffs[..., l4_slice]) + 1e-12)
        ).item()

    model = _build_model()
    train_ds = SyntheticDataset("0e + 2e", num_samples=512, teacher=teacher, seed=0)
    test_ds = SyntheticDataset("0e + 2e", num_samples=128, teacher=teacher, seed=1)

    x_test, _, _, true_A_test, _ = test_ds.generate()
    initial_error = _l4_error(model, x_test, true_A_test)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    x_train, y_train, _, _, _ = train_ds.generate()
    for _ in range(100):
        data = _make_data_object(x_train)
        result = model(data, target=y_train, return_scale=False)
        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()

    final_error = _l4_error(model, x_test, true_A_test)
    assert final_error < initial_error

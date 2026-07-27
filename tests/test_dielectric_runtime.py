from __future__ import annotations

import pytest
import torch
from e3nn import o3

from scripts.dielectric_runtime import (
    DielectricRunSpec,
    build_dielectric_model,
    load_run_spec,
    write_run_spec,
)
from tests.test_equivariance import _build_graph, _rotate_data


def _spec(**overrides):
    values = dict(
        hidden_dim=8,
        lmax=2,
        num_layers=1,
        num_basis=4,
        atom_features="manual",
        tp_backend="e3nn",
        cueq_method="naive",
        covariance_parameterization="centered_spectral_window",
        log_variance_min=-4.0,
        log_variance_max=4.0,
        shape_min=-2.0,
        shape_max=2.0,
        volume_min=-8.0,
        volume_max=8.0,
        distribution="student_t",
        student_t_dof=5.0,
        representation_metric="block_auto",
        metric_scalar=2.0,
        metric_l2=3.0,
    )
    values.update(overrides)
    return DielectricRunSpec(**values)


def test_run_spec_round_trip(tmp_path):
    spec = _spec()
    write_run_spec(
        tmp_path,
        spec,
        compilation={"compiler": "test"},
        training_stage="mean",
        init_checkpoint=None,
    )
    assert load_run_spec(tmp_path) == spec


def test_run_spec_is_strict(tmp_path):
    with pytest.raises(FileNotFoundError, match="explicit migration"):
        load_run_spec(tmp_path)


def test_centered_contract_requires_metric_and_builds_cpu():
    spec = _spec(metric_scalar=None)
    with pytest.raises(ValueError, match="metric_scalar"):
        build_dielectric_model(spec, "cpu")


def test_factory_model_is_end_to_end_o3_equivariant():
    """The production RunSpec factory preserves mean and scatter equivariance."""
    torch.manual_seed(7)
    spec = _spec(
        num_basis=8,
        representation_metric="none",
        metric_scalar=None,
        metric_l2=None,
    )
    model, _ = build_dielectric_model(spec, "cpu")
    model.eval()
    data = _build_graph(torch.randn(7, 3), num_basis=8)
    with torch.no_grad():
        output = model(data, return_scale=True)
    # Cover both a proper rotation and an improper O(3) transformation.  The
    # latter is essential because the compiler tracks parity, not SO(3) alone.
    transformations = (o3.rand_matrix(), torch.diag(torch.tensor([-1.0, 1.0, 1.0])))
    for matrix in transformations:
        transformed_data = _rotate_data(data, matrix)
        with torch.no_grad():
            transformed = model(transformed_data, return_scale=True)
        rho = model.output_spec.representation_matrix(matrix)
        torch.testing.assert_close(
            transformed["mu"], output["mu"] @ rho.T, atol=3e-5, rtol=3e-5
        )
        expected_scale = rho @ output["scale"] @ rho.T
        torch.testing.assert_close(
            transformed["scale"], expected_scale, atol=3e-4, rtol=3e-4
        )


def test_compiled_faithful_path_does_not_train_shared_lifting_from_covariance():
    torch.manual_seed(9)
    spec = _spec(
        num_basis=8,
        representation_metric="none",
        metric_scalar=None,
        metric_l2=None,
    )
    model, _ = build_dielectric_model(spec, "cpu")
    data = _build_graph(torch.randn(7, 3), num_basis=8)
    target = torch.randn(1, 6)
    features, graph_batch = model.backbone(data)
    faithful = model.forward_faithful_from_features(features, graph_batch, target=target)
    faithful["loss"].backward()
    lifting_grad = torch.cat(
        [p.grad.reshape(-1) for p in model.joint_head.lifting.parameters() if p.grad is not None]
    )
    assert torch.isfinite(lifting_grad).all() and lifting_grad.norm() > 0

    model.zero_grad(set_to_none=True)
    features_mean, graph_batch_mean = model.backbone(data)
    mean_result = model.forward_from_features(features_mean, graph_batch_mean, target=target)
    torch.nn.functional.mse_loss(mean_result["mu"], target).backward()
    mean_grad = torch.cat(
        [p.grad.reshape(-1) for p in model.joint_head.lifting.parameters() if p.grad is not None]
    )
    torch.testing.assert_close(lifting_grad, mean_grad, atol=1e-6, rtol=1e-5)

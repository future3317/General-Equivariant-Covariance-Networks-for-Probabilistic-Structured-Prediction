import numpy as np
import pytest
import torch
from argparse import Namespace

from compatibility.e3nn import o3
from data.elasticity_normalization import ElasticityTargetNormalizer
from representations import rank4_elasticity_irreps
from scripts.train_elasticity import _configure_arm
from scripts.train_elasticity import train_epoch


def _normalizer() -> ElasticityTargetNormalizer:
    values = np.random.default_rng(7).normal(size=(32, 21))
    return ElasticityTargetNormalizer.fit(
        values, mode="representation_compatible"
    )


def test_representation_compatible_normalization_commutes_with_o3_action():
    normalizer = _normalizer()
    values = torch.randn(4, 21)
    rotation = o3.rand_matrix()
    irreps = o3.Irreps(rank4_elasticity_irreps())

    unnormalized = normalizer.transform(values)
    physical_rotated = normalizer.inverse(
        unnormalized @ irreps.D_from_matrix(rotation).T
    )
    lhs = normalizer.transform(physical_rotated)
    rhs = unnormalized @ irreps.D_from_matrix(rotation).T
    torch.testing.assert_close(lhs, rhs, atol=2e-5, rtol=2e-5)


def test_representation_compatible_normalization_round_trips_physical_targets():
    normalizer = _normalizer()
    values = torch.randn(5, 21)
    recovered = normalizer.inverse(normalizer.transform(values))
    torch.testing.assert_close(recovered, values, atol=2e-5, rtol=2e-5)


def test_legacy_normalization_remains_available_for_reproducibility():
    normalizer = ElasticityTargetNormalizer.fit(
        np.random.default_rng(8).normal(size=(32, 21)), mode="legacy_voigt"
    )
    values = torch.randn(3, 21)
    recovered = normalizer.inverse(normalizer.transform(values))
    torch.testing.assert_close(recovered, values, atol=2e-5, rtol=2e-5)


def test_named_elasticity_arms_preserve_minimal_study_contract():
    args = Namespace(arm="deterministic", objective="gaussian", covariance="auto")
    _configure_arm(args)
    assert args.objective == "deterministic"
    assert args.covariance is None

    args = Namespace(arm="full_student_t", objective="gaussian", covariance="auto")
    _configure_arm(args)
    assert args.objective == "student_t"
    assert args.covariance == "full"


class _Batch:
    def __init__(self, loss):
        self.edge_index = torch.tensor([[0], [0]])
        self.y_irreps = torch.zeros(1, 1)
        self.loss = loss

    def to(self, device, non_blocking=False):
        return self


class _NonfiniteLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, batch, *, target, return_scale):
        return {"loss": self.weight * batch.loss}


def test_elasticity_training_rejects_nonfinite_loss_before_optimizer_step():
    model = _NonfiniteLossModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(FloatingPointError, match="non-finite elasticity training loss"):
        train_epoch(
            model,
            [_Batch(torch.tensor(float("nan")))],
            optimizer,
            torch.device("cpu"),
        )

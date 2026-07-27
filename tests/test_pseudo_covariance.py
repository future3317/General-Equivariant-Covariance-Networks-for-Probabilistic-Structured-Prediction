from __future__ import annotations

import pytest
import torch

from data.pseudo_covariance import (
    PSEUDO_CACHE_VERSION,
    build_isotropic_pseudo_covariance,
    invariant_structure_embedding,
    validate_oof_residual_payload,
    validate_pseudo_cache,
)
from data.oof import fold_assignments


class _Graph:
    def __init__(self, pos: torch.Tensor, features: torch.Tensor):
        self.pos = pos
        self.node_features = features


def test_invariant_embedding_is_o3_translation_and_permutation_invariant():
    torch.manual_seed(4)
    pos = torch.randn(7, 3, dtype=torch.float64)
    features = torch.randn(7, 5, dtype=torch.float64)
    reference = invariant_structure_embedding(_Graph(pos, features))
    proper = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64)).Q
    improper = proper @ torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    for transform in (proper, improper):
        order = torch.randperm(pos.shape[0])
        changed = _Graph(pos[order] @ transform.T + torch.tensor([3.0, -2.0, 1.0]), features[order])
        torch.testing.assert_close(invariant_structure_embedding(changed), reference, atol=1e-11, rtol=1e-11)


def test_isotropic_pseudolabel_is_safe_and_weights_decrease_with_distance():
    residuals = torch.tensor([[0., 0.], [1., 0.], [0., 2.], [3., 1.]], dtype=torch.float64)
    embeddings = torch.tensor([[0.], [0.1], [1.0], [3.0]], dtype=torch.float64)
    result = build_isotropic_pseudo_covariance(residuals, embeddings, k=2, tau=0.25, shrinkage=0.2, epsilon=1e-9)
    covariance = result["covariance"]
    torch.testing.assert_close(covariance, covariance.transpose(-1, -2))
    torch.testing.assert_close(covariance[:, 0, 0], covariance[:, 1, 1])
    torch.testing.assert_close(covariance[:, 0, 1], torch.zeros(4, dtype=torch.float64))
    assert torch.all(result["weights"] > 0)
    assert torch.all(result["effective_neighbours"] >= 1.0)
    # For query 0, neighbour 1 is closer than neighbour 2 and gets more mass.
    order = result["neighbours"][0]
    assert result["weights"][0, (order == 1).nonzero().item()] > result["weights"][0, (order == 2).nonzero().item()]


def test_directional_cache_is_hard_rejected_and_oof_leakage_is_rejected():
    with pytest.raises(ValueError, match="train split"):
        validate_oof_residual_payload({"split": "test", "folds": 5, "residuals": torch.zeros(2, 2), "fold_assignments": torch.zeros(2, dtype=torch.long)})
    bad = {
        "version": PSEUDO_CACHE_VERSION, "split": "train", "mode": "transported_full",
        "coordinate_semantics": "residual_covariance", "transport_certificate": None,
        "covariance": torch.eye(2)[None], "sqrt_covariance": torch.eye(2)[None],
        "isotropic_variance": torch.ones(1), "neighbours": torch.zeros(1, 1, dtype=torch.long), "weights": torch.ones(1, 1),
    }
    with pytest.raises(ValueError, match="directional pseudo-labels"):
        validate_pseudo_cache(bad)


def test_untransported_directional_knn_fails_single_query_rotation_gate():
    """The forbidden construction cannot be mistaken for an O(3) target.

    Rotating only a query changes the required output covariance frame while
    its invariant-neighbour residual pool remains unchanged.  A generic
    anisotropic local covariance therefore violates equivariance.
    """
    residuals = torch.tensor([[1., 0.], [0., 2.], [3., 1.]], dtype=torch.float64)
    embeddings = torch.tensor([[0.], [1.], [2.]], dtype=torch.float64)
    result = build_isotropic_pseudo_covariance(residuals, embeddings, k=2, tau=10.0, shrinkage=0.0, epsilon=1e-8)
    unsafe_directional = result["raw_residual_covariance"][0]
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    assert not torch.allclose(unsafe_directional, rotation @ unsafe_directional @ rotation.T)
    # The executable isotropic projection, in contrast, commutes with every
    # orthogonal coordinate action.
    safe = result["covariance"][0]
    torch.testing.assert_close(safe, rotation @ safe @ rotation.T)


def test_oof_assignment_is_deterministic_and_each_holdout_is_disjoint():
    assignments = fold_assignments(101, folds=5, seed=17)
    torch.testing.assert_close(assignments, fold_assignments(101, folds=5, seed=17))
    for fold in range(5):
        train = torch.where(assignments != fold)[0]
        held_out = torch.where(assignments == fold)[0]
        assert torch.isin(held_out, train).sum() == 0
        assert len(train) + len(held_out) == 101

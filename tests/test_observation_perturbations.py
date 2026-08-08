import numpy as np

from data.observation_perturbations import (
    DepthPerturbationScale,
    perturb_depth_observation,
)


def test_depth_perturbations_are_seeded_and_input_only():
    depth = np.full((20, 30), 2.0, dtype=np.float32)
    depth[0, 0] = 0.0
    scale = DepthPerturbationScale(missing_fraction=0.1, depth_noise_std=0.01)
    for kind in ("missing_block", "point_dropout", "depth_noise"):
        left = perturb_depth_observation(
            depth,
            kind=kind,
            scale=scale,
            rng=np.random.default_rng(7),
        )
        right = perturb_depth_observation(
            depth,
            kind=kind,
            scale=scale,
            rng=np.random.default_rng(7),
        )
        assert np.array_equal(left, right)
        assert left.shape == depth.shape
        assert left[0, 0] == 0.0


def test_missing_block_uses_requested_area_scale():
    depth = np.ones((100, 100), dtype=np.float32)
    perturbed = perturb_depth_observation(
        depth,
        kind="missing_block",
        scale=DepthPerturbationScale(
            missing_fraction=0.04,
            depth_noise_std=0.001,
        ),
        rng=np.random.default_rng(3),
    )
    assert np.isclose((perturbed == 0.0).mean(), 0.04)

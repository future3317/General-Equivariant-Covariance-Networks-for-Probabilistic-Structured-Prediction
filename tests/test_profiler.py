"""Smoke tests for the profiler script."""

from pathlib import Path

from scripts.profile_dielectric import count_tp_instructions


def test_count_tp_instructions_runs():
    """count_tp_instructions should run without error on a simple model."""
    from representations import O3IrrepsSpec
    from models import (
        EquivariantBackbone,
        EquivariantMeanHead,
        O3QuadraticSymmetricOperatorHead,
        StructuredProbabilisticPredictor,
    )
    from spd_maps import MatrixExponentialMap
    from distributions import GaussianNLL

    output_spec = O3IrrepsSpec("0e + 2e")
    backbone = EquivariantBackbone(
        hidden_dim=8, lmax=1, num_layers=1, atom_feature_dim=49, num_basis=4,
    )
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

    total, instructions = count_tp_instructions(model)
    assert total >= 0
    assert isinstance(instructions, list)


def test_profiler_script_imports():
    """Profiler script should import without errors."""
    import scripts.profile_dielectric as profile_mod
    assert Path(profile_mod.__file__).name == "profile_dielectric.py"

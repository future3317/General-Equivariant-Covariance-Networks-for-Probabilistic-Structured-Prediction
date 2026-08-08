import torch

from evaluation.elliptical import (
    elliptical_falsification_from_whitened,
    falsification_decision,
)
from scripts.audit_elliptical_law import _tertiles


def _student_t_samples(
    count: int, dimension: int, dof: float, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    normal = torch.randn(count, dimension, generator=generator, dtype=torch.float64)
    chi2 = torch.distributions.Chi2(dof).sample((count,)).double()
    return normal * torch.sqrt(torch.tensor(dof) / chi2).unsqueeze(-1)


def test_matching_student_t_is_not_strongly_rejected():
    torch.manual_seed(4)
    samples = _student_t_samples(6000, 4, 5.0, 4)
    audit = elliptical_falsification_from_whitened(
        samples,
        reference="student_t",
        student_t_dof=5.0,
        num_directions=24,
        permutations=49,
        seed=3,
    )
    assert audit["radial_pit"]["ks"] < 0.04
    assert audit["projection_pit"]["median_ks"] < 0.035
    assert audit["direction_sphericality"]["second_moment_defect"] < 0.18
    assert audit["radius_direction_dependence"]["max_abs_spearman"] < 0.08
    assert not falsification_decision(audit)[
        "single_component_elliptical_structure_rejected"
    ]


def test_directional_misspecification_is_detected():
    torch.manual_seed(7)
    samples = _student_t_samples(5000, 4, 5.0, 7)
    samples[:, 0] *= 2.5
    audit = elliptical_falsification_from_whitened(
        samples,
        reference="student_t",
        student_t_dof=5.0,
        num_directions=24,
        permutations=49,
        seed=5,
    )
    assert audit["projection_pit"]["max_ks"] > 0.08
    assert audit["direction_sphericality"]["second_moment_defect"] > 0.5
    assert falsification_decision(audit)[
        "single_component_elliptical_structure_rejected"
    ]


def test_tied_descriptor_quantiles_produce_nonempty_semantic_strata():
    values = torch.tensor([2.0] * 4 + [3.0] * 16 + [4.0] * 5 + [5.0])
    strata = _tertiles(values, "element_count")
    assert list(strata) == [
        "element_count_below_tied_cut",
        "element_count_at_tied_cut",
        "element_count_above_tied_cut",
    ]
    assert [int(mask.sum()) for mask in strata.values()] == [4, 16, 6]
    assert torch.stack(list(strata.values())).sum(dim=0).eq(1).all()

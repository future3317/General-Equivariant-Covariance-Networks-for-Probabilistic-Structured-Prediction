from scripts.audit_itop_reviewer_controls import _rotation_summary


def test_rotation_summary_reports_rotation_and_reflection_counts():
    records = [
        {
            "equivariance_error": 0.1,
            "nll_delta": 0.2,
            "coverage90_delta": -0.01,
            "coverage95_delta": 0.03,
            "is_reflection": False,
        },
        {
            "equivariance_error": 0.4,
            "nll_delta": -0.6,
            "coverage90_delta": 0.02,
            "coverage95_delta": -0.04,
            "is_reflection": True,
        },
    ]
    summary = _rotation_summary(records, rotations=2)
    assert summary["transform_count"] == 2
    assert summary["rotation_count"] == 1
    assert summary["reflection_count"] == 1
    assert summary["equivariance_error_max"] == 0.4
    assert summary["nll_delta_max_abs"] == 0.6
    assert summary["all_finite"] is True

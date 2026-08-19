import json

import torch

from scripts.audit_elasticity_artifacts import audit_campaign


def test_asinh_campaign_audit_accepts_complete_contract(tmp_path):
    for seed in (42, 43, 44):
        run = tmp_path / f"seed_{seed}" / "full_asinh_exp_student_t_reprnorm"
        run.mkdir(parents=True)
        (run / "args.json").write_text(
            json.dumps(
                {
                    "arm": "full_asinh_exp_student_t",
                    "objective": "student_t",
                    "student_t_dof": 5.0,
                    "seed": seed,
                    "target_normalization": "representation_compatible",
                }
            )
        )
        (run / "compilation.json").write_text(
            json.dumps(
                {
                    "family": {"kind": "asinh_exponential"},
                    "representation_reachability": {
                        "active": {
                            "reachable": True,
                            "depth": 3,
                            "target_irreps": "1x8e",
                        },
                        "canonical": {"reachable": True, "depth": 3},
                    },
                }
            )
        )
        (run / "metrics.json").write_text(
            json.dumps(
                {
                    "finite": True,
                    "nll": 1.0,
                    "fp64_scatter": {"strict_spd": True, "minimum_eigenvalue": 0.1},
                }
            )
        )
        (run / "history.json").write_text(
            json.dumps([{"epoch": 1, "val_loss": 1.0}])
        )
        for name in ("environment.json", "best_model.pt", "train.log"):
            if name.endswith(".json"):
                (run / name).write_text("{}")
            elif name.endswith(".pt"):
                torch.save({}, run / name)
            else:
                (run / name).write_text("validation selection\n")
        torch.save({"mean": torch.ones(2)}, run / "predictions.pt")

    report = audit_campaign(tmp_path)
    assert report["eligible"] is True
    assert [run["active_depth"] for run in report["runs"]] == [3, 3, 3]

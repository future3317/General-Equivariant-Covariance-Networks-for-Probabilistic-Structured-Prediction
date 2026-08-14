from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.frozen_operator_arm import (
    FrozenOperatorArmSpec,
    evaluate_frozen_operator_arm,
    train_frozen_operator_arm,
)
from models.frozen_distribution_readout import FrozenMeanScatterElliptical
from spd_maps import IsotropicMap


def _records(seed: int, count: int = 24) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(count, 2, generator=generator)
    mean = torch.randn(count, 3, generator=generator)
    target = mean + 0.3 * torch.randn(count, 3, generator=generator)
    return [
        {
            "features": features[index],
            "mean": mean[index],
            "target": target[index],
            "sample_id": torch.tensor(index),
        }
        for index in range(count)
    ]


def test_frozen_operator_arm_selects_only_by_validation_and_writes_checkpoints(
    tmp_path: Path,
):
    train_records = _records(3)
    validation_loader = DataLoader(_records(4), batch_size=8, shuffle=False)
    model = FrozenMeanScatterElliptical(
        "2x0e",
        "0e",
        IsotropicMap(dim=3),
        distribution="student_t",
        student_t_dof=5.0,
    )
    spec = FrozenOperatorArmSpec(
        run_dir=tmp_path,
        seed=42,
        max_epochs=4,
        patience=2,
        learning_rate=5e-4,
        weight_decay=1e-5,
    )

    def train_loader_for_epoch(epoch: int):
        generator = torch.Generator().manual_seed(42 + epoch)
        return DataLoader(
            train_records, batch_size=8, shuffle=True, generator=generator
        )

    result = train_frozen_operator_arm(
        model,
        train_loader_for_epoch=train_loader_for_epoch,
        validation_loader=validation_loader,
        device=torch.device("cpu"),
        spec=spec,
        checkpoint_metadata={"family": "isotropic", "law": "student_t"},
    )

    selected = min(result["history"], key=lambda row: row["validation_nll"])
    assert result["selection_split"] == "validation"
    assert result["selected_epoch"] == selected["epoch"]
    assert (tmp_path / "history.json").is_file()
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "last_model.pt").is_file()


def test_frozen_operator_evaluation_preserves_frozen_artifacts():
    records = _records(7, count=24)
    loader = DataLoader(records, batch_size=5, shuffle=False)
    model = FrozenMeanScatterElliptical(
        "2x0e",
        "0e",
        IsotropicMap(dim=3),
        distribution="gaussian",
    )
    metrics, predictions = evaluate_frozen_operator_arm(
        model,
        loader,
        device=torch.device("cpu"),
        distribution="gaussian",
        student_t_dof=5.0,
        seed=11,
    )

    assert metrics["nll_semantics"] == "exact_single_gaussian_log_likelihood"
    assert torch.isfinite(torch.tensor(metrics["nll"]))
    torch.testing.assert_close(
        predictions["mean"], torch.stack([record["mean"] for record in records])
    )
    torch.testing.assert_close(
        predictions["target"],
        torch.stack([record["target"] for record in records]),
    )
    torch.testing.assert_close(
        predictions["sample_id"],
        torch.stack([record["sample_id"] for record in records]),
    )
    assert predictions["scale"].shape == (24, 3, 3)
    assert bool(torch.isfinite(predictions["scale"]).all())

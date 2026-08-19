from __future__ import annotations

from scripts.complete_elasticity_artifacts import _history_from_log


def test_history_recovery_reads_validation_selection_from_training_log(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "2026-01-01 INFO - Epoch 1/2: train_loss=2.0, train_nll=1.9, "
        "val_loss=3.0, val_mae=4.0\n"
        "2026-01-01 INFO - Epoch 2/2: train_loss=1.0, train_nll=0.9, "
        "val_loss=2.0, val_mae=3.0\n",
        encoding="utf-8",
    )

    history = _history_from_log(log)

    assert [row["epoch"] for row in history] == [1, 2]
    assert min(history, key=lambda row: row["val_loss"])["epoch"] == 2

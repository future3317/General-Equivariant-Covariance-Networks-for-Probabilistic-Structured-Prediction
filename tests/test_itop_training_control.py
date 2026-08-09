"""Regression tests for ITOP optimization, stopping, and exact resume state."""

from __future__ import annotations

import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from scripts.evaluate_itop_final import _available_models
from scripts.itop_reproducibility import training_contract
from scripts.run_itop_e3a import _member_command
from scripts.run_itop_study import (
    GEOMETRY_CACHE_FILES,
    SELECTABLE_PROBABILISTIC_MODELS,
    _geometry_cache_complete,
    _model_list,
    _parse_args,
    _training_command,
)
from scripts.train_itop import (
    MODEL_KINDS,
    _build_model,
    _capture_rng_state,
    _configure_initialization,
    _freeze_record,
    _load_checkpoint,
    _materialize_evaluation_scatter,
    _restore_rng_state,
    _save_checkpoint,
    _set_loader_epoch,
    _test_loader_kwargs,
    _update_early_stopping,
    train_epoch,
)
from spd_maps import MatrixExponentialMap


class _NonFiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class _TinyCachedFeatureModel(torch.nn.Module):
    def __init__(self, failure: str | None = None):
        super().__init__()
        self.backbone = torch.nn.Identity()
        self.readout = torch.nn.Linear(3, 45)
        self.distribution = object()
        self.failure = failure

    def forward_from_features(
        self,
        features,
        graph_batch,
        *,
        target,
        return_scale,
    ):
        del graph_batch, return_scale
        mean = self.readout(features)
        loss = torch.nn.functional.mse_loss(mean, target)
        if self.failure == "loss":
            loss = loss * loss.new_tensor(float("nan"))
        elif self.failure == "gradient":
            loss = _NonFiniteGradient.apply(loss)
        return {
            "mu": mean,
            "loss": loss,
            "components": {
                "loss_fit": loss.detach() * 0.75,
                "loss_uncertainty": loss.detach() * 0.25,
            },
        }


def _cached_feature_batches():
    return [
        {
            "features": torch.randn(4, 3),
            "target": torch.randn(4, 45),
        },
        {
            "features": torch.randn(3, 3),
            "target": torch.randn(3, 45),
        },
    ]


def test_train_epoch_records_components_and_gradient_norms():
    model = _TinyCachedFeatureModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    statistics = train_epoch(
        model,
        _cached_feature_batches(),
        optimizer,
        torch.device("cpu"),
        frozen_backbone=False,
        use_bf16=False,
    )
    assert set(statistics) == {
        "loss",
        "gradient_norm_mean",
        "gradient_norm_max",
        "loss_fit",
        "loss_uncertainty",
    }
    assert all(np.isfinite(value) for value in statistics.values())
    assert statistics["gradient_norm_max"] >= statistics["gradient_norm_mean"]


@pytest.mark.parametrize("failure", ("loss", "gradient"))
def test_train_epoch_fails_fast_on_nonfinite_optimization(failure):
    model = _TinyCachedFeatureModel(failure=failure)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises((FloatingPointError, RuntimeError), match="non-finite"):
        train_epoch(
            model,
            _cached_feature_batches(),
            optimizer,
            torch.device("cpu"),
            frozen_backbone=False,
            use_bf16=False,
        )


def test_early_stopping_tracks_improvement_and_rejects_nan():
    best, stale = float("inf"), 0
    best, stale, improved = _update_early_stopping(3.0, best, stale)
    assert (best, stale, improved) == (3.0, 0, True)
    best, stale, improved = _update_early_stopping(3.1, best, stale)
    assert (best, stale, improved) == (3.0, 1, False)
    best, stale, improved = _update_early_stopping(2.9, best, stale)
    assert (best, stale, improved) == (2.9, 0, True)
    with pytest.raises(FloatingPointError, match="validation criterion"):
        _update_early_stopping(float("nan"), best, stale)


def test_itop_evaluation_materializes_frozen_generator_in_fp64_without_repair():
    params = torch.diag_embed(torch.tensor([[0.0, -4.0, 3.0]], dtype=torch.float32))
    scale, certificate = _materialize_evaluation_scatter(MatrixExponentialMap(), params)
    assert scale.dtype == torch.float64
    assert certificate["dtype"] == "float64"
    assert certificate["policy"] == "reject_if_not_strict"
    assert certificate["strict"] is True
    torch.testing.assert_close(
        scale,
        torch.diag_embed(torch.exp(params.double().diagonal(dim1=-2, dim2=-1))),
    )


def test_training_sample_order_is_seed_and_epoch_addressable():
    dataset = TensorDataset(torch.arange(32))
    sampler = RandomSampler(dataset, generator=torch.Generator())
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)

    _set_loader_epoch(loader, seed=42, epoch=7)
    first = torch.cat([batch[0] for batch in loader])
    torch.rand(100)
    _set_loader_epoch(loader, seed=42, epoch=7)
    resumed = torch.cat([batch[0] for batch in loader])
    _set_loader_epoch(loader, seed=42, epoch=8)
    next_epoch = torch.cat([batch[0] for batch in loader])

    torch.testing.assert_close(resumed, first, rtol=0.0, atol=0.0)
    assert not torch.equal(next_epoch, first)


def test_existing_run_directory_requires_resumable_state(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = SimpleNamespace(
        data_dir=tmp_path / "ITOP",
        num_points=256,
        batch_size=4,
        num_workers=0,
        train_cache_sample_limit=None,
    )
    with pytest.raises(FileExistsError, match="no resumable state"):
        _training_command(
            args,
            run_dir=run_dir,
            model="deterministic",
            phase="deterministic",
            seed=42,
            epochs=2,
        )

    (run_dir / "last_state.pt").touch()
    command = _training_command(
        args,
        run_dir=run_dir,
        model="deterministic",
        phase="deterministic",
        seed=42,
        epochs=2,
    )
    assert command[-1] == "--continue_run"
    assert command[command.index("--tp_backend") + 1] == "e3nn"


def test_test_loader_kwargs_exclude_train_cache_limit():
    kwargs = _test_loader_kwargs(
        {"train_cache_sample_limit": 2487, "batch_size": 16}
    )
    assert kwargs == {"batch_size": 16}


def test_geometry_cache_requires_exact_sample_limit(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "metadata.json").write_text('{"sample_limit": 64}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale ITOP geometry cache"):
        _geometry_cache_complete(cache, sample_limit=None)
    for name in GEOMETRY_CACHE_FILES:
        (cache / name).touch()
    assert _geometry_cache_complete(cache, sample_limit=64)


def test_runner_exposes_explicit_joint_skip_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_itop_study",
            "--data_dir",
            "data",
            "--study_dir",
            "results",
            "--gpu",
            "3",
            "--skip_joint_finetune",
        ],
    )
    assert _parse_args().skip_joint_finetune is True


def test_runner_accepts_a_minimal_full_student_t_pilot(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_itop_study",
            "--data_dir",
            "data",
            "--study_dir",
            "results",
            "--gpu",
            "2",
            "--split_seed",
            "42",
            "--models",
            "full_student_t",
            "--joint_models",
            "full_student_t",
            "--deterministic_epochs",
            "8",
            "--frozen_epochs",
            "5",
            "--joint_epochs",
            "3",
            "--patience",
            "2",
        ],
    )
    args = _parse_args()
    assert _model_list(args.models, option="--models") == ("full_student_t",)
    assert _model_list(args.joint_models, option="--joint_models") == (
        "full_student_t",
    )
    assert (args.deterministic_epochs, args.frozen_epochs, args.joint_epochs) == (
        8,
        5,
        3,
    )
    assert args.split_seed == 42
    assert "full_student_t" in SELECTABLE_PROBABILISTIC_MODELS


def test_study_training_command_forwards_shared_split_seed(tmp_path):
    args = SimpleNamespace(
        data_dir=tmp_path / "ITOP",
        num_points=256,
        batch_size=4,
        num_workers=0,
        train_cache_sample_limit=2487,
        split_seed=42,
    )
    command = _training_command(
        args,
        run_dir=tmp_path / "run",
        model="deterministic",
        phase="deterministic",
        seed=43,
        epochs=2,
    )
    assert command[command.index("--seed") + 1] == "43"
    assert command[command.index("--split_seed") + 1] == "42"


def test_e3a_member_command_keeps_model_and_split_seeds_distinct(tmp_path):
    args = SimpleNamespace(
        data_dir=tmp_path / "ITOP",
        split_seed=42,
        batch_size=16,
        num_workers=8,
        num_epochs=5,
        tp_backend="e3nn",
        cueq_method="naive",
    )
    command = _member_command(args, tmp_path / "member", seed=44)
    assert command[command.index("--seed") + 1] == "44"
    assert command[command.index("--split_seed") + 1] == "42"


def test_training_contract_records_effective_split_seed():
    args = SimpleNamespace(
        model="deterministic",
        phase="deterministic",
        student_t_dof=5.0,
        representation_metric="none",
        seed=43,
        split_seed=42,
        num_points=256,
        num_neighbors=16,
        train_cache_sample_limit=2487,
        lr=5e-4,
        weight_decay=1e-5,
        batch_size=16,
        patience=2,
        backbone_precision="bf16",
        num_workers=8,
        tp_backend="e3nn",
        cueq_method="naive",
        compile_tp=False,
    )
    contract = training_contract(args, torch.device("cpu"), freeze={})
    assert contract["data"]["split_seed"] == 42
    assert contract["randomness"]["seed"] == 43


def test_runner_rejects_unknown_or_repeated_model_filters():
    with pytest.raises(ValueError, match="unsupported"):
        _model_list("full_student_t,not_a_model", option="--models")
    with pytest.raises(ValueError, match="distinct"):
        _model_list("full_student_t,full_student_t", option="--models")


def test_itop_factorial_exposes_radial_and_operator_controls():
    assert "independent_student_t" in MODEL_KINDS
    assert "low_rank_student_t" in MODEL_KINDS
    assert "full_student_t" in MODEL_KINDS


def test_end_to_end_phase_requires_independent_probabilistic_initialization():
    model = torch.nn.Linear(2, 2)
    args = SimpleNamespace(
        phase="end_to_end",
        model="full_student_t",
        backbone_checkpoint=None,
        resume_checkpoint=None,
    )
    assert _configure_initialization(model, args) is False
    args.resume_checkpoint = "old.pt"
    with pytest.raises(ValueError, match="independent initialization"):
        _configure_initialization(model, args)


def test_end_to_end_freeze_contract_allows_only_bypassed_mean_projection():
    model, _ = _build_model(
        SimpleNamespace(
            model="full_student_t",
            hidden_dim=16,
            max_radius=0.5,
            lmax=2,
            num_layers=1,
            num_basis=4,
            tp_backend="e3nn",
            cueq_method="naive",
            student_t_dof=5.0,
        )
    )
    freeze = _freeze_record(
        model,
        SimpleNamespace(
            phase="end_to_end",
            model="full_student_t",
            backbone_checkpoint=None,
            resume_checkpoint=None,
        ),
    )
    assert freeze["frozen_parameter_names"] == [
        "joint_head.operator_head.mean_projection.weight"
    ]


def test_joint_freeze_contract_allows_only_bypassed_mean_projection():
    model, _ = _build_model(
        SimpleNamespace(
            model="full_student_t",
            hidden_dim=16,
            max_radius=0.5,
            lmax=2,
            num_layers=1,
            num_basis=4,
            tp_backend="e3nn",
            cueq_method="naive",
            student_t_dof=5.0,
        )
    )
    freeze = _freeze_record(
        model,
        SimpleNamespace(
            phase="joint_finetune",
            model="full_student_t",
            backbone_checkpoint=None,
            resume_checkpoint=None,
        ),
    )
    assert freeze["frozen_parameter_names"] == [
        "joint_head.operator_head.mean_projection.weight"
    ]
    assert freeze["boundary"].startswith("all active parameters trainable")


@pytest.mark.parametrize(
    ("model", "covariance_mode"),
    (("independent_student_t", "graph"), ("low_rank_student_t", "low_rank")),
)
def test_itop_factorial_models_bind_to_distinct_compiler_families(
    model, covariance_mode
):
    _, plan = _build_model(
        SimpleNamespace(
            model=model,
            hidden_dim=16,
            max_radius=0.5,
            lmax=2,
            num_layers=1,
            num_basis=4,
            tp_backend="e3nn",
            cueq_method="naive",
            student_t_dof=5.0,
        )
    )
    assert plan.compilation.covariance_mode == covariance_mode
    assert plan.compilation.distribution_spec.objective_name() == "student_t"


def test_itop_full_control_binds_to_unrestricted_family():
    _, plan = _build_model(
        SimpleNamespace(
            model="full_student_t",
            hidden_dim=16,
            max_radius=0.5,
            lmax=2,
            num_layers=1,
            num_basis=4,
            tp_backend="e3nn",
            cueq_method="naive",
            student_t_dof=5.0,
        )
    )
    assert plan.compilation.covariance_mode == "full"
    assert plan.compilation.covariance_parameter_count == 1035


def test_frozen_head_contract_records_only_backbone_and_mean_head():
    args = SimpleNamespace(
        model="graph_student_t",
        phase="frozen_head",
        backbone_checkpoint=None,
        resume_checkpoint=None,
        hidden_dim=16,
        max_radius=0.5,
        lmax=2,
        num_layers=1,
        num_basis=4,
        tp_backend="e3nn",
        cueq_method="naive",
        student_t_dof=5.0,
    )
    model, _ = _build_model(args)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in model.joint_head.mean_head.parameters():
        parameter.requires_grad_(False)
    record = _freeze_record(model, args)
    assert record["boundary"].startswith("backbone and direct mean head frozen")
    assert record["frozen_parameter_count"] > 0
    assert record["trainable_parameter_count"] > 0
    assert all(
        name.startswith(
            (
                "backbone.",
                "joint_head.mean_head.",
                "joint_head.operator_head.mean_projection.",
            )
        )
        for name in record["frozen_parameter_names"]
    )


def test_final_evaluator_ignores_incomplete_stopped_ablation(tmp_path):
    root = tmp_path / "seed_42"
    complete = root / "frozen_graph_student_t"
    partial = root / "joint_graph_student_t"
    complete.mkdir(parents=True)
    partial.mkdir(parents=True)
    for filename in (
        "metrics.json",
        "history.json",
        "predictions_side.pt",
        "predictions_top.pt",
        "args.json",
        "environment.json",
        "train.log",
    ):
        (complete / filename).touch()
    (partial / "history.json").touch()
    (partial / "best_model.pt").touch()
    assert _available_models(root) == ("frozen_graph_student_t",)


def test_rng_checkpoint_round_trip_is_exact(tmp_path):
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = _capture_rng_state()
    checkpoint = tmp_path / "last_state.pt"
    _save_checkpoint({"rng_state": state}, checkpoint)

    expected = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restored = _load_checkpoint(checkpoint)
    _restore_rng_state(restored["rng_state"])
    actual = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )

    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)

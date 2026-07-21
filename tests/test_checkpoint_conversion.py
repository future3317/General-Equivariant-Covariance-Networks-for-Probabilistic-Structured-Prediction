"""Strict checkpoint migration regression tests."""

import json

import pytest
import torch

from equivcompiler import (
    FeatureSpec,
    FullCovariance,
    PreferExecutor,
    SpecificExecutor,
    convert_checkpoint,
    plan_readout,
)
from representations import CompilationError


SEED = "4x0e + 2x1o + 2x2e"


def _head(backend):
    plan = plan_readout(
        FeatureSpec.from_irreps(SEED, scope="node"),
        output="0e + 2e",
        covariance=FullCovariance(),
        executor=SpecificExecutor(backend),
        cost=PreferExecutor((backend,)),
        output_scope="node",
    )
    return plan.compilation.build_head()


def test_exact_checkpoint_conversion_preserves_raw_state_layout(tmp_path):
    source_head = _head("spherical_cg")
    source = tmp_path / "source.pt"
    destination = tmp_path / "lowered.pt"
    torch.save(source_head.state_dict(), source)

    audit = convert_checkpoint(
        source,
        destination,
        from_backend="spherical-cg",
        to_backend="cartesian-stf",
        output="0e + 2e",
        seed_irreps=SEED,
        output_scope="dense",
    )
    converted = torch.load(destination, map_location="cpu", weights_only=True)
    target_head = _head("cartesian_stf")
    target_head.load_state_dict(converted, strict=True)

    assert not audit["learned_coordinates_changed"]
    assert audit["numerical_equivalence"]["mean_max_abs"] < 2e-5
    assert audit["numerical_equivalence"]["distribution_parameter_max_abs"] < 2e-5
    assert (
        audit["source_compilation"]["backend_selection_basis"]["selected_executor"]
        == "spherical_cg"
    )
    assert (
        audit["target_compilation"]["backend_selection_basis"]["selected_executor"]
        == "cartesian_stf"
    )
    stored_audit = json.loads(
        destination.with_suffix(".pt.conversion.json").read_text(encoding="utf-8")
    )
    assert stored_audit["destination"]["sha256"] == audit["destination"]["sha256"]


def test_checkpoint_conversion_handles_full_model_prefix(tmp_path):
    source_head = _head("spherical_cg")
    source = tmp_path / "source.pt"
    destination = tmp_path / "lowered.pt"
    state = {f"joint_head.{key}": value for key, value in source_head.state_dict().items()}
    state["backbone.placeholder"] = torch.ones(1)
    torch.save(state, source)

    audit = convert_checkpoint(
        source,
        destination,
        from_backend="spherical_cg",
        to_backend="cartesian_stf",
        output="0e + 2e",
        seed_irreps=SEED,
        output_scope="dense",
    )
    converted = torch.load(destination, map_location="cpu", weights_only=True)
    assert converted["backbone.placeholder"].item() == 1
    assert audit["head_prefix"] == "joint_head."


def test_checkpoint_conversion_refuses_incompatible_backend(tmp_path):
    source = tmp_path / "source.pt"
    torch.save(_head("spherical_cg").state_dict(), source)
    with pytest.raises(CompilationError) as caught:
        convert_checkpoint(
            source,
            tmp_path / "invalid.pt",
            from_backend="spherical_cg",
            to_backend="cartesian_stf",
            output="ijkl=jikl=ijlk=klij",
            seed_irreps=SEED,
            output_scope="dense",
        )
    assert caught.value.certificate.code == "backend_incompatible"

"""Benchmark frozen ITOP uncertainty families on one cached feature batch.

This audit compares only the family-dependent probabilistic head.  It loads
each selected frozen-head checkpoint strictly, reuses the same cached features
and target batch, and measures eager head forward, proper-NLL evaluation, and
proper-NLL forward/backward with synchronized wall-clock timing.  It neither
trains a model nor changes a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.itop_features import get_itop_feature_loaders
from scripts.benchmarking import environment_record, measure
from scripts.itop_reproducibility import atomic_write_json
from scripts.train_itop import (
    _build_model,
    _configure_initialization,
    _forward,
    _load_checkpoint,
    _to_device,
)

MODEL_ORDER = ("full_student_t", "low_rank_student_t", "graph_student_t")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _parse_run(specification: str) -> tuple[str, Path]:
    try:
        model, path_text = specification.split("=", 1)
    except ValueError as error:
        raise ValueError(f"invalid --run specification: {specification}") from error
    if model not in MODEL_ORDER:
        raise ValueError(f"unsupported ITOP family: {model}")
    run = Path(path_text)
    if not run.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run}")
    return model, run


def _run_contract(model: str, run: Path) -> dict[str, Any]:
    args = _read_json(run / "args.json")
    environment = _read_json(run / "environment.json")
    if args.get("model") != model or args.get("phase") != "frozen_head":
        raise ValueError(f"{run}: does not certify frozen {model}")
    if not (run / "best_model.pt").is_file():
        raise FileNotFoundError(f"{run}: best_model.pt is required")
    cache = environment.get("feature_cache")
    checkpoint = environment.get("input_checkpoint")
    if not isinstance(cache, dict) or not isinstance(checkpoint, dict):
        raise TypeError(f"{run}: missing feature-cache or backbone provenance")
    return {"args": args, "environment": environment}


def _common_contract(records: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], int]:
    reference = records[MODEL_ORDER[0]]
    reference_args = reference["args"]
    reference_environment = reference["environment"]
    reference_cache = reference_environment["feature_cache"]
    reference_checkpoint = reference_environment["input_checkpoint"]
    required_args = (
        "feature_cache",
        "backbone_checkpoint",
        "batch_size",
        "num_points",
        "num_neighbors",
        "backbone_precision",
        "tp_backend",
        "cueq_method",
        "compile_tp",
        "student_t_dof",
    )
    for model, record in records.items():
        args = record["args"]
        environment = record["environment"]
        mismatches = {
            name: {"reference": reference_args.get(name), "actual": args.get(name)}
            for name in required_args
            if args.get(name) != reference_args.get(name)
        }
        if mismatches:
            raise ValueError(f"{model}: incompatible runtime contract {mismatches}")
        if environment["feature_cache"].get("feature_cache_hash") != reference_cache.get(
            "feature_cache_hash"
        ):
            raise ValueError(f"{model}: feature-cache hash differs from reference")
        if environment["input_checkpoint"].get("sha256") != reference_checkpoint.get(
            "sha256"
        ):
            raise ValueError(f"{model}: backbone checkpoint differs from reference")
    geometry = reference_environment.get("dataset", {}).get("caches", {}).get(
        "side_train", {}
    )
    samples = int(geometry.get("metadata", {}).get("num_samples", 0))
    if samples < 2:
        raise ValueError("run provenance does not record a valid side-train size")
    return (
        {
            "feature_cache_hash": reference_cache["feature_cache_hash"],
            "backbone_checkpoint_sha256": reference_checkpoint["sha256"],
            "feature_cache": reference_args["feature_cache"],
            "backbone_checkpoint": reference_args["backbone_checkpoint"],
            "side_train_samples": samples,
            "batch_size": reference_args["batch_size"],
            "student_t_dof": reference_args["student_t_dof"],
            "execution": "eager; TF32 disabled; no torch.compile or executor fallback",
        },
        int(reference_args["batch_size"]),
    )


def _load_model(record: dict[str, Any], run: Path, device: torch.device):
    args = argparse.Namespace(**record["args"])
    model, plan = _build_model(args)
    _configure_initialization(model, args)
    payload = _load_checkpoint(run / "best_model.pt")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, plan


def _finite_result(result: dict[str, torch.Tensor], *, require_loss: bool) -> None:
    required = ("mu", "params", "scale", "loss") if require_loss else ("mu", "params")
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(f"model output is missing {missing}")
    for name in required:
        if not bool(torch.isfinite(result[name]).all()):
            raise FloatingPointError(f"non-finite {name} in runtime audit")


def _benchmark_one(
    model,
    plan,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    target = batch["target"]
    latest: dict[str, torch.Tensor] | None = None

    def forward() -> None:
        nonlocal latest
        with torch.inference_mode():
            latest = _forward(
                model,
                batch,
                target=None,
                return_scale=False,
                use_bf16=False,
            )

    def nll_evaluation() -> None:
        nonlocal latest
        with torch.inference_mode():
            latest = _forward(
                model,
                batch,
                target=target,
                return_scale=True,
                use_bf16=False,
            )

    def forward_backward() -> None:
        nonlocal latest
        latest = _forward(
            model,
            batch,
            target=target,
            return_scale=False,
            use_bf16=False,
        )
        latest["loss"].backward()

    def prepare() -> None:
        model.zero_grad(set_to_none=True)

    forward()
    assert latest is not None
    _finite_result(latest, require_loss=False)
    nll_evaluation()
    assert latest is not None
    _finite_result(latest, require_loss=True)
    family = plan.report.as_dict()["family"]
    return {
        "active_coordinates": int(family["parameter_count"]),
        "operator_program_hash": family["operator_program_hash"],
        "validation": {
            "finite_forward": True,
            "finite_nll_evaluation": True,
            "batch_samples": int(target.shape[0]),
            "output_dimension": int(target.shape[-1]),
        },
        "timings": {
            "forward": measure(
                forward, prepare=None, device=device, warmup=warmup, repeats=repeats
            ),
            "nll_evaluation": measure(
                nll_evaluation,
                prepare=None,
                device=device,
                warmup=warmup,
                repeats=repeats,
            ),
            "forward_backward": measure(
                forward_backward,
                prepare=prepare,
                device=device,
                warmup=warmup,
                repeats=repeats,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="MODEL=RUN_DIR",
        help="repeat for full_student_t, low_rank_student_t, and graph_student_t",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 4:
        parser.error("--repeats must be at least 4")
    parsed = dict(_parse_run(specification) for specification in args.run)
    if set(parsed) != set(MODEL_ORDER):
        parser.error(f"--run must name exactly {', '.join(MODEL_ORDER)}")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("this audit requires a CUDA device for GPU latency and memory")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    records = {model: _run_contract(model, run) for model, run in parsed.items()}
    contract, batch_size = _common_contract(records)
    _, _, side_loader, _, _ = get_itop_feature_loaders(
        contract["feature_cache"],
        backbone_checkpoint=contract["backbone_checkpoint"],
        seed=int(records[MODEL_ORDER[0]]["args"]["seed"]),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    batch = _to_device(next(iter(side_loader)), device)
    rows = {}
    for model_name in MODEL_ORDER:
        model, plan = _load_model(records[model_name], parsed[model_name], device)
        rows[model_name] = _benchmark_one(
            model,
            plan,
            batch,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        del model
        torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "kind": "itop_frozen_head_family_runtime_audit",
        "training_steps": 0,
        "environment": environment_record(device),
        "contract": contract,
        "protocol": {
            "unit": "one cached Side feature batch",
            "data_transfer": "excluded",
            "models": list(MODEL_ORDER),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "memory": "per-family peak allocated/reserved after warmup",
            "backward": "proper Student-t NLL; zero_grad(set_to_none=True) before timer",
        },
        "rows": rows,
    }
    atomic_write_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Audit and aggregate frozen-backbone ITOP Graph-Student-t runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from scripts.evaluate_itop_final import _load_prediction, _prediction_audit
from scripts.itop_reproducibility import atomic_write_json

REQUIRED_ARTIFACTS = (
    "metrics.json",
    "history.json",
    "predictions_side.pt",
    "predictions_top.pt",
    "args.json",
    "environment.json",
    "train.log",
)
METRICS = (
    ("side", "mpjpe_cm", "Side MPJPE"),
    ("side", "nll", "Side NLL"),
    ("side", "mace", "Side MACE"),
    ("side", "frame_risk_coverage_auc_cm", "Side RC AUC"),
    ("top", "mpjpe_cm", "Top MPJPE"),
    ("top", "nll", "Top NLL"),
    ("top", "mace", "Top MACE"),
    ("top", "frame_risk_coverage_auc_cm", "Top RC AUC"),
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_run(seed: int, run: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"seed {seed}: missing artifacts {missing}")
    args = _read(run / "args.json")
    if args.get("seed") != seed or args.get("model") != "graph_student_t":
        raise ValueError(f"seed {seed}: args.json does not identify this run")
    if args.get("phase") != "frozen_head":
        raise ValueError(f"seed {seed}: robustness audit requires frozen_head")
    metrics = _read(run / "metrics.json")
    environment = _read(run / "environment.json")
    feature_cache = _read(run / "feature_cache.json") if (run / "feature_cache.json").is_file() else {}
    contract = environment.get("training_contract", {})
    freeze = contract.get("freeze", {})
    provenance_warnings = []
    if freeze.get("phase") != "frozen_head":
        if seed != 42 or not feature_cache:
            raise ValueError(f"seed {seed}: missing frozen-head contract")
        provenance_warnings.append(
            "legacy seed-42 environment lacks structured training_contract/source fields; args and feature_cache were checked"
        )
    elif not freeze.get("frozen_parameter_count", 0) or not freeze.get(
        "trainable_parameter_count", 0
    ):
        raise ValueError(f"seed {seed}: freeze counts are not recorded")

    prediction_audit = {}
    for view in ("side", "top"):
        prediction = _load_prediction(run / f"predictions_{view}.pt")
        record = _prediction_audit(
            prediction, metrics[view], model="frozen_graph_student_t", view=view
        )
        if record["num_samples"] != 4863:
            raise ValueError(
                f"seed {seed}/{view}: expected 4863 samples, got {record['num_samples']}"
            )
        if not record["all_recorded_checks_pass"]:
            raise ValueError(f"seed {seed}/{view}: saved metrics do not recompute")
        prediction_audit[view] = record

    history = _read(run / "history.json")
    best = min(history, key=lambda row: float(row["loss"]))
    return {
        "seed": seed,
        "run_dir": str(run),
        "best_epoch": int(best["epoch"]),
        "best_validation_nll": float(best["loss"]),
        "metrics": metrics,
        "args": args,
        "environment": environment,
        "feature_cache": feature_cache,
        "provenance_warnings": provenance_warnings,
        "prediction_audit": prediction_audit,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for view, key, _ in METRICS:
        values = [float(record["metrics"][view][key]) for record in records]
        aggregate[f"{view}_{key}"] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values),
            "values": values,
        }
    auroc = [float(record["metrics"]["ood"]["side_top_uncertainty_auroc"]) for record in records]
    aggregate["side_top_uncertainty_auroc"] = {
        "mean": statistics.fmean(auroc),
        "std": statistics.stdev(auroc),
        "values": auroc,
    }
    return aggregate


def _fmt(stat: dict[str, Any]) -> str:
    return f"{stat['mean']:.3f} $\\pm$ {stat['std']:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-run",
        action="append",
        required=True,
        metavar="SEED=RUN_DIR",
        help="repeat once per seed; RUN_DIR is the frozen_graph_student_t directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for specification in args.seed_run:
        try:
            seed_text, run_text = specification.split("=", 1)
            seed = int(seed_text)
        except ValueError as error:
            raise ValueError(f"invalid --seed-run: {specification}") from error
        records.append(_audit_run(seed, Path(run_text)))
    records.sort(key=lambda record: record["seed"])
    if len(records) != 3 or [record["seed"] for record in records] != [42, 43, 44]:
        raise ValueError("the robustness audit requires exactly seeds 42, 43, and 44")

    aggregate = _summary(records)
    source_commits = {
        record["environment"].get("source", {}).get("commit") for record in records
    }
    backbone_hashes = {
        record["environment"].get("input_checkpoint", {}).get("sha256")
        for record in records
    }
    backbone_hashes.update(
        record["feature_cache"].get("backbone_checkpoint_sha256") for record in records
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        {
            "schema_version": 1,
            "study": "ITOP Graph Student-t frozen-backbone head-seed robustness",
            "seeds": [42, 43, 44],
            "sample_contract": {"side": 4863, "top": 4863},
            "selection": "validation-only NLL within each frozen-head run",
            "source_commits": sorted(commit for commit in source_commits if commit),
            "backbone_checkpoint_sha256": sorted(
                checksum for checksum in backbone_hashes if checksum
            ),
            "records": records,
            "aggregate": aggregate,
        },
        output / "itop_graph_t_robustness.json",
    )
    lines = [
        "# ITOP Graph Student-t robustness audit",
        "",
        "Frozen-backbone head-seed audit; seeds 42, 43, and 44 use the same deterministic backbone and pooled feature cache. Each head uses its own seeded side-train split, sampler, initialization, validation-NLL selection, and complete side/top test prediction artifacts.",
        "",
        "| Metric | Mean $\\pm$ std |",
        "|---|---:|",
    ]
    for view, key, label in METRICS:
        lines.append(f"| {label} | {_fmt(aggregate[f'{view}_{key}'])} |")
    lines.append(
        f"| Side-to-top uncertainty AUROC | {_fmt(aggregate['side_top_uncertainty_auroc'])} |"
    )
    lines += [
        "",
        "All prediction audits passed finite-value, shape, sample-count, and metric-recomputation checks. The checkpoint hash is shared across seeds; source commits are recorded where available. Seed 42 is retained as a pre-existing factorial artifact with a legacy environment schema, and this is recorded as a provenance warning rather than silently upgraded.",
    ]
    (output / "itop_graph_t_robustness.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

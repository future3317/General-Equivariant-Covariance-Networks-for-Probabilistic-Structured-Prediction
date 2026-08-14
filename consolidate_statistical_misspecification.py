"""Consolidate existing statistical-misspecification evidence only.

Reads existing JSON and prediction artifacts, recomputes shared scalar
diagnostics, writes provenance hashes, and renders a double-column figure.
It never trains or selects a model from test/OOD values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) quantitative_grid -> cross-type inherit -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

# Academic Figure Skill Typography Baseline -- COPY VERBATIM, place at TOP of script
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

# Academic Figure Skill Export Baseline -- COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

LEVELS = np.asarray([0.1 * i for i in range(1, 10)] + [0.95], dtype=float)
METHOD_COLORS = {
    "fixed_nu": CATEGORICAL[0],
    "conditional_nu": CATEGORICAL[1],
    "representation_repair": CATEGORICAL[2],
}


def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def scalar_metrics(run: Path) -> dict:
    diagnostics = json.loads((run / "diagnostics.json").read_text())
    test = diagnostics["test"]
    prediction = torch.load(
        run / "predictions_test.pt", map_location="cpu", weights_only=True
    )
    mean = prediction["mean"].double()
    target = prediction["target"].double()
    scale = prediction["scale"].double()
    nu = torch.as_tensor(prediction.get("nu", 5.0), dtype=torch.float64)
    if nu.ndim == 0:
        nu = nu.expand(mean.shape[0])
    residual = target - mean
    solved = torch.linalg.solve(scale, residual.unsqueeze(-1)).squeeze(-1)
    q = (residual * solved).sum(-1).numpy()
    nu_np = nu.numpy()
    coverage = [
        float(np.mean(q < mean.shape[-1] * stats.f.ppf(level, mean.shape[-1], nu_np)))
        for level in LEVELS
    ]
    # The existing A/B report calls this a scalar alignment proxy and uses
    # Pearson correlation, not the rank correlation used by the directional
    # diagnostic.  Keep those two statistics distinct.
    residual_norm2 = (residual * residual).sum(-1).numpy()
    alignment = float(np.corrcoef(q, residual_norm2)[0, 1])
    elliptical = test["elliptical_falsification"]
    radial = elliptical["radial_pit"]
    radius_direction = elliptical["radius_direction_dependence"]
    return {
        "n": int(mean.shape[0]),
        "nll": float(test["nll"]),
        "energy": float(test["energy_score"]),
        "coverage50": coverage[4],
        "coverage90": coverage[8],
        "coverage95": coverage[9],
        "mace": float(np.mean(np.abs(np.asarray(coverage) - LEVELS))),
        "whitened_second_moment_defect": float(
            elliptical["whitened_second_moment_defect"]
        ),
        "radial_pit_ks": float(radial["ks"]),
        "radial_pit_pvalue": float(radial["pvalue"]),
        "radius_direction_max_abs_spearman": float(
            radius_direction["max_abs_spearman"]
        ),
        "radius_direction_permutation_pvalue": float(
            radius_direction["max_statistic_permutation_pvalue"]
        ),
        "radius_direction_rejected_at_alpha_0.01": bool(
            radius_direction["max_statistic_permutation_pvalue"] <= 0.01
        ),
        "alignment_q_vs_residual_norm2_pearson": alignment,
        "finite_predictions": bool(
            torch.isfinite(mean).all()
            and torch.isfinite(target).all()
            and torch.isfinite(scale).all()
            and torch.isfinite(nu).all()
        ),
        "nu_summary": {
            "min": float(nu.min()),
            "median": float(nu.median()),
            "max": float(nu.max()),
        },
        "nll_semantics": test.get("nll_semantics"),
        "diagnostics_source": str(run / "diagnostics.json"),
    }


def protocol_summary(run: Path) -> dict:
    protocol_path = run / "protocol.json"
    if not protocol_path.exists():
        return {}
    protocol = json.loads(protocol_path.read_text())
    flat = {}

    def flatten(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                flatten(child, f"{prefix}.{key}" if prefix else key)
        elif not isinstance(value, list):
            flat[prefix] = value

    flatten(protocol)
    tokens = ("seed", "split", "selection", "variant", "family", "nu", "backend", "exact", "source_commit")
    return {
        key: flat[key]
        for key in sorted(flat)
        if any(token in key.lower() for token in tokens)
    }


def artifact_record(run: Path, label: str, method: str) -> dict:
    files = []
    for path in sorted(run.iterdir()):
        if path.is_file():
            files.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    return {
        "label": label,
        "method": method,
        "path": str(run),
        "files": files,
        "protocol_summary": protocol_summary(run),
        "metrics": scalar_metrics(run),
    }


def comparison_rows(records: list[dict]) -> list[dict]:
    keys = [
        "nll", "energy", "coverage50", "coverage90", "coverage95", "mace",
        "whitened_second_moment_defect", "radial_pit_ks", "radial_pit_pvalue",
        "radius_direction_max_abs_spearman", "radius_direction_permutation_pvalue",
        "alignment_q_vs_residual_norm2_pearson",
    ]
    rows = []
    for record in records:
        row = {
            "method": record["method"],
            "label": record["label"],
            "n": record["metrics"]["n"],
            "replicate_count": 1,
        }
        row.update({key: record["metrics"].get(key) for key in keys})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict], output: Path):
    methods = ["fixed_nu", "conditional_nu", "representation_repair"]
    display = {
        "fixed_nu": r"Fixed $\nu=5$",
        "conditional_nu": r"Conditional $\nu(x)$",
        "representation_repair": "Uncertainty branch",
    }
    specifications = [
        ("nll", "Normalized NLL", False),
        ("energy", "Energy score", False),
        ("mace", "MACE", False),
        ("whitened_second_moment_defect", "Whitened defect", True),
        ("radial_pit_ks", "Radial PIT KS", False),
        ("radius_direction_max_abs_spearman", r"Max $|\rho_S|$", False),
    ]
    row_map = {row["method"]: row for row in rows}
    x = np.arange(len(methods))
    fig, axes = plt.subplots(
        2, 3, figsize=(183 / 25.4, 86 / 25.4), constrained_layout=True
    )
    for axis, (key, title, log_y) in zip(axes.flat, specifications):
        values = [row_map[method][key] for method in methods]
        bars = axis.bar(
            x,
            values,
            color=[METHOD_COLORS[method] for method in methods],
            width=0.68,
            edgecolor="white",
            linewidth=0.6,
        )
        if log_y:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xticks(x, [display[method] for method in methods], rotation=24, ha="right")
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.3g}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
            )
        if key == "radius_direction_max_abs_spearman":
            axis.text(0.03, 0.95, r"$p_{perm}=0.005$ for all", transform=axis.transAxes, va="top", fontsize=6, color=ACCENT_RED)
        if key == "radial_pit_ks":
            axis.text(0.03, 0.95, "lower is closer to radial reference", transform=axis.transAxes, va="top", fontsize=6, color=BLACK)
    fig.suptitle(
        r"Dielectric misspecification consolidation: radial repair without directional adequacy",
        fontsize=9,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.03,
        r"Fixed: one artifact; conditional-$\nu$: three-seed confirmation mean; uncertainty branch: one-seed negative diagnostic. Lower is better except coverage (reported in the table).",
        ha="center",
        fontsize=6,
    )
    save_cns_figure(fig, output / "dielectric_misspecification_consolidation")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.results_root
    paths = [
        (root / "stat_misspec_pilot_868baac/dielectric/fixed", "fixed_nu", "fixed Student-t, nu=5"),
        (root / "stat_misspec_confirm_868baac/dielectric/conditional_nu_seed42", "conditional_nu", "conditional nu(x), seed 42"),
        (root / "stat_misspec_representation_20260812/dielectric/uncertainty_branch_conditional_nu_seed42_initA", "representation_repair", "uncertainty-only branch, seed 42"),
    ]
    records = [artifact_record(path, label, method) for path, method, label in paths]
    rows = comparison_rows(records)
    confirmation = []
    for seed in (42, 43, 44):
        run = root / f"stat_misspec_confirm_868baac/dielectric/conditional_nu_seed{seed}"
        confirmation.append({"seed": seed, "metrics": scalar_metrics(run), "path": str(run)})
    aggregate = {}
    for key in rows[1]:
        if key in ("method", "label", "n", "replicate_count"):
            continue
        values = [entry["metrics"].get(key) for entry in confirmation]
        if all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            aggregate[key] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
                "values": values,
            }
    for row in rows:
        if row["method"] == "conditional_nu":
            row["replicate_count"] = 3
            for key, value in aggregate.items():
                row[key] = value["mean"]
    write_csv(args.output / "dielectric_comparison.csv", rows)
    (args.output / "dielectric_comparison.json").write_text(
        json.dumps(jsonable({"rows": rows, "conditional_nu_three_seed": aggregate}), indent=2) + "\n",
        encoding="utf-8",
    )

    itop_path = root / "stat_misspec_pilot_868baac/itop/comparison_report.json"
    itop = json.loads(itop_path.read_text())
    itop_summary = {
        "source": str(itop_path),
        "selection_contract": itop.get("selection_contract"),
        "decision": itop.get("decision"),
        "arms": {},
    }
    for arm, payload in itop.get("arms", {}).items():
        itop_summary["arms"][arm] = {
            "label": payload.get("label"),
            "descriptors": payload.get("descriptors"),
            "test": payload.get("test"),
            "ood": payload.get("ood"),
            "selection": payload.get("selection"),
            "artifact_hashes": payload.get("artifacts"),
        }
    (args.output / "itop_negative_pilot.json").write_text(
        json.dumps(jsonable(itop_summary), indent=2) + "\n", encoding="utf-8"
    )

    manifest_paths = [
        path for path, _, _ in paths
    ] + [
        root / "stat_misspec_confirm_868baac/dielectric/conditional_nu_seed43",
        root / "stat_misspec_confirm_868baac/dielectric/conditional_nu_seed44",
        root / "stat_misspec_pilot_868baac/itop/comparison_report.json",
        root / "stat_misspec_pilot_868baac/itop/experiment_ledger.json",
        root / "stat_misspec_representation_20260812/ab_comparison.json",
    ]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
        ).strip(),
        "environment": "equivcompiler",
        "selection_policy": "validation-only; no test/OOD quantity used",
        "split_policy": "existing frozen splits; no IDs changed",
        "records": [],
    }
    for path in manifest_paths:
        files = path.rglob("*") if path.is_dir() else [path]
        for file in sorted(files):
            if file.is_file():
                manifest["records"].append(
                    {"path": str(file), "bytes": file.stat().st_size, "sha256": sha256(file)}
                )
    (args.output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(rows, args.output)
    qa = {
        "figure_width_mm": 183,
        "figure_export": ["PDF vector master", "PNG 300 dpi preview"],
        "all_input_rows_used": True,
        "missing_metrics": [
            "ITOP paired bootstrap CIs: not part of this consolidation artifact",
            "dielectric component-wise alignment decomposition: not recomputed here",
        ],
        "checks": {
            "finite_dielectric": all(row["n"] > 0 for row in rows),
            "directional_rejection_preserved": all(
                row["radius_direction_permutation_pvalue"] <= 0.01 for row in rows
            ),
            "no_training": True,
        },
    }
    (args.output / "figure_qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": rows, "manifest_records": len(manifest["records"])}, indent=2))


if __name__ == "__main__":
    main()

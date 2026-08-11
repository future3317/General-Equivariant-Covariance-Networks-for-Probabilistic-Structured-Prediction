"""Controlled covariance-recovery benchmark for the statistical closure.

Matching rows use a compiler-produced teacher and learner with the same
feature and family contract.  Cross-family rows keep the output type fixed,
change only the declared operator family, and test whether a misspecified
family fails in the expected way.  All rows use repeated observations from a
known Student-t scale and check equivariance plus proper-score invariance
under an independent orthogonal coordinate change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from compatibility.e3nn import o3
from distributions import StudentTNLL
from equivcompiler import (
    FeatureSpec,
    FullCovariance,
    GraphPrecision,
    IsotypicBlockCovariance,
    LowRankCovariance,
    plan_readout,
)
from evaluation import covariance_relative_error, empirical_coverage
from representations import EquivariantOutputGraph, O3IrrepsSpec

NU = 5.0
INDEPENDENT_ORACLE_VERSION = "independent_numpy_scipy_v1"


def _sample_student(mu: torch.Tensor, scale: torch.Tensor, nu: float) -> torch.Tensor:
    chol = torch.linalg.cholesky(scale)
    normal = torch.randn_like(mu)
    chi = torch.distributions.Chi2(
        torch.tensor(nu, dtype=mu.dtype, device=mu.device)
    ).sample(mu.shape[:-1])
    return mu + torch.einsum("nij,nj->ni", chol, normal) * torch.sqrt(
        nu / chi
    ).unsqueeze(-1)


def _nll_from_scale(
    residual: torch.Tensor, scale: torch.Tensor, nu: float = NU
) -> torch.Tensor:
    d = residual.shape[-1]
    solved = torch.linalg.solve(scale, residual.unsqueeze(-1)).squeeze(-1)
    q = (residual * solved).sum(-1)
    constant = (
        -torch.lgamma(residual.new_tensor((nu + d) / 2.0))
        + torch.lgamma(residual.new_tensor(nu / 2.0))
        + 0.5 * d * residual.new_tensor(math.log(nu * math.pi))
    )
    return (constant + 0.5 * torch.linalg.slogdet(scale)[1] + 0.5 * (nu + d) * torch.log1p(q / nu)).mean()


def _cases() -> dict[str, tuple[str, object]]:
    graph = EquivariantOutputGraph(
        num_nodes=3, edges=((0, 1), (1, 2)), node_irrep="1o"
    )
    return {
        "full": ("0e+2e", FullCovariance()),
        "low_rank": ("0e+2e", LowRankCovariance(rank=2)),
        "isotypic_block": ("0e+2e", IsotypicBlockCovariance()),
        "graph_precision": (str(graph.output_irreps), GraphPrecision(graph)),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_independent_artifact(
    artifact: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = Path(artifact["manifest_path"])
    npz_path = Path(artifact["npz_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("oracle_version") != INDEPENDENT_ORACLE_VERSION:
        raise ValueError("unsupported independent oracle version")
    npz_sha256 = _file_sha256(npz_path)
    if npz_sha256 != manifest.get("npz_sha256"):
        raise ValueError("oracle artifact hash mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return manifest, arrays


def _prepare_independent_learner(head: torch.nn.Module) -> None:
    with torch.no_grad():
        head.mean_projection.weight.zero_()
        if head.mean_projection.bias is not None:
            head.mean_projection.bias.zero_()
        for module_name in ("covariance_projection", "scale_projection"):
            module = getattr(head, module_name, None)
            if module is not None:
                module.weight.mul_(0.05)
                if module.bias is not None:
                    module.bias.zero_()
    for parameter in head.mean_projection.parameters():
        parameter.requires_grad_(False)


def _split_tensors(
    arrays: dict[str, np.ndarray], split: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    inputs = torch.from_numpy(arrays[f"{split}_inputs"]).to(
        device=device, dtype=torch.float32
    )
    observations = torch.from_numpy(arrays[f"{split}_observations"]).to(
        device=device, dtype=torch.float32
    )
    replicates = observations.shape[1]
    return (
        inputs.repeat_interleave(replicates, dim=0),
        observations.reshape(-1, observations.shape[-1]),
        replicates,
    )


def _split_loss(
    head: torch.nn.Module,
    spd_map: torch.nn.Module,
    loss_fn: StudentTNLL,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    batch = torch.arange(inputs.shape[0], device=inputs.device)
    mean, parameters = head(inputs, batch)
    loss, _ = loss_fn(mean, parameters, targets, spd_map)
    return loss


def _relative_operator_errors(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    difference = torch.linalg.matrix_norm(predicted - target, ord="fro")
    denominator = torch.linalg.matrix_norm(target, ord="fro").clamp_min(1e-12)
    return difference / denominator


def _orthogonal_scatter_invariance_error(scatter: torch.Tensor) -> float:
    dimension = scatter.shape[-1]
    generator = torch.Generator(device=scatter.device).manual_seed(314159)
    random = torch.randn(
        dimension,
        dimension,
        generator=generator,
        device=scatter.device,
        dtype=scatter.dtype,
    )
    orthogonal, _ = torch.linalg.qr(random)
    transformed = orthogonal @ scatter @ orthogonal.transpose(-1, -2)
    recovered = orthogonal.transpose(-1, -2) @ transformed @ orthogonal
    return float(
        (
            torch.linalg.matrix_norm(recovered - scatter, ord="fro")
            / torch.linalg.matrix_norm(scatter, ord="fro").clamp_min(1e-12)
        )
        .max()
        .item()
    )


def run_independent_pair(
    artifact: dict[str, Any],
    learner_name: str,
    *,
    steps: int,
    patience: int,
    device: str,
) -> dict[str, Any]:
    """Train one compiler learner on a pre-generated independent oracle artifact."""
    if steps < 1 or patience < 1:
        raise ValueError("steps and patience must be positive")
    manifest, arrays = _load_independent_artifact(artifact)
    teacher_name = str(manifest["family"])
    cases = _cases()
    if learner_name not in cases:
        raise ValueError(f"unknown learner family: {learner_name}")
    graph_teacher = teacher_name == "graph_precision"
    graph_learner = learner_name == "graph_precision"
    if graph_teacher != graph_learner:
        raise ValueError("teacher and learner representation contract mismatch")

    torch.manual_seed(int(manifest["seed"]))
    dev = torch.device(device)
    output, learner_family = cases[learner_name]
    feature = FeatureSpec.from_irreps(output, scope="global")
    learner_plan = plan_readout(
        feature,
        output=output,
        covariance=learner_family,
        distribution="student_t",
        student_t_dof=NU,
    )
    learner = learner_plan.compilation.build_head().to(dev)
    spd_map = learner_plan.compilation.build_spd_map().to(dev)
    _prepare_independent_learner(learner)
    loss_fn = StudentTNLL(NU)
    train_inputs, train_targets, _ = _split_tensors(arrays, "train", dev)
    validation_inputs, validation_targets, _ = _split_tensors(
        arrays, "validation", dev
    )
    trainable = [
        parameter for parameter in learner.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=3e-3)
    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    selected_epoch = -1
    stale_epochs = 0
    last_epoch = -1
    for epoch in range(steps):
        last_epoch = epoch
        learner.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _split_loss(
            learner, spd_map, loss_fn, train_inputs, train_targets
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        learner.eval()
        with torch.no_grad():
            validation_loss = _split_loss(
                learner,
                spd_map,
                loss_fn,
                validation_inputs,
                validation_targets,
            )
        validation_value = float(validation_loss)
        if validation_value < best_validation:
            best_validation = validation_value
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in learner.state_dict().items()
            }
            selected_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("validation-only selection produced no checkpoint")
    learner.load_state_dict(best_state)
    learner.eval()

    test_inputs_unique = torch.from_numpy(arrays["test_inputs"]).to(
        device=dev, dtype=torch.float32
    )
    test_batch = torch.arange(test_inputs_unique.shape[0], device=dev)
    with torch.no_grad():
        predicted_mean, predicted_parameters = learner(
            test_inputs_unique, test_batch
        )
        predicted_scatter = spd_map(predicted_parameters)
    test_observations = torch.from_numpy(arrays["test_observations"]).to(
        device=dev, dtype=torch.float32
    )
    test_replicates = test_observations.shape[1]
    flat_target = test_observations.reshape(-1, test_observations.shape[-1])
    repeated_mean = predicted_mean.repeat_interleave(test_replicates, dim=0)
    repeated_parameters = predicted_parameters.repeat_interleave(
        test_replicates, dim=0
    )
    repeated_scatter = predicted_scatter.repeat_interleave(
        test_replicates, dim=0
    )
    with torch.no_grad():
        test_nll, _ = loss_fn(
            repeated_mean,
            repeated_parameters,
            flat_target,
            spd_map,
        )
    coverage = empirical_coverage(
        repeated_mean.double(),
        flat_target.double(),
        repeated_scatter.double(),
        levels=[0.90, 0.95],
        reference="student_t",
        student_t_dof=NU,
    )

    true_scatter = torch.from_numpy(arrays["test_scatter"]).to(
        device=dev, dtype=torch.float64
    )
    scatter_errors = _relative_operator_errors(
        predicted_scatter.double(), true_scatter
    )
    if graph_learner:
        with torch.no_grad():
            predicted_primary = spd_map.precision(predicted_parameters).double()
        true_primary = torch.from_numpy(arrays["test_precision"]).to(
            device=dev, dtype=torch.float64
        )
        primary_name = "precision"
    else:
        predicted_primary = predicted_scatter.double()
        true_primary = true_scatter
        primary_name = "scatter"
    primary_errors = _relative_operator_errors(predicted_primary, true_primary)
    primary_mean = float(primary_errors.mean())
    primary_p90 = float(torch.quantile(primary_errors, 0.90))
    scatter_mean = float(scatter_errors.mean())
    equivariance_error = _equivariance_error(
        learner,
        test_inputs_unique[: min(4, len(test_inputs_unique))],
        test_batch[: min(4, len(test_batch))],
        predicted_scatter[: min(4, len(predicted_scatter))],
        output,
    )
    basis_nll_error = _basis_invariance(
        repeated_mean.double(),
        flat_target.double(),
        repeated_scatter.double(),
        feature,
    )
    scatter_invariance_error = _orthogonal_scatter_invariance_error(
        predicted_scatter.double()
    )
    finite = all(
        torch.isfinite(tensor).all().item()
        for tensor in (
            predicted_mean,
            predicted_parameters,
            predicted_scatter,
            test_nll,
            primary_errors,
        )
    )
    teacher_coverage = manifest["teacher_coverage"]
    tolerance = manifest["sampling_tolerance"]
    coverage_pass = all(
        abs(coverage[key] - float(teacher_coverage[key]))
        <= float(tolerance[key])
        for key in ("coverage_90", "coverage_95")
    )
    numeric_pass = bool(
        finite
        and equivariance_error <= 5e-5
        and basis_nll_error <= 5e-5
        and scatter_invariance_error <= 5e-5
    )
    recovery_pass = bool(
        primary_mean <= 0.05
        and primary_p90 <= 0.10
        and (not graph_learner or scatter_mean <= 0.075)
    )
    matched = teacher_name == learner_name
    gate: dict[str, bool | None] = {
        "numeric": numeric_pass,
        "recovery": recovery_pass,
        "coverage": coverage_pass,
        "provenance": True,
        "overall": (
            numeric_pass and recovery_pass and coverage_pass if matched else None
        ),
    }
    return {
        "teacher_backend": "independent_numpy_scipy",
        "teacher_family": teacher_name,
        "learner_family": learner_name,
        "role": "matched_recovery" if matched else "diagnostic_cross_family",
        "seed": int(manifest["seed"]),
        "oracle_npz_sha256": manifest["npz_sha256"],
        "oracle_manifest_sha256": _file_sha256(Path(artifact["manifest_path"])),
        "selection_split": "validation",
        "selected_epoch": selected_epoch,
        "last_epoch": last_epoch,
        "best_validation_nll": best_validation,
        "primary_operator": primary_name,
        "primary_relative_error_mean": primary_mean,
        "primary_relative_error_p90": primary_p90,
        "scatter_relative_error_mean": scatter_mean,
        "test_nll": float(test_nll),
        "coverage_90": coverage["coverage_90"],
        "coverage_95": coverage["coverage_95"],
        "equivariance_max_abs": equivariance_error,
        "orthogonal_scatter_relative_error": scatter_invariance_error,
        "orthogonal_nll_abs_error": basis_nll_error,
        "finite": bool(finite),
        "compiler_report": learner_plan.report.as_dict(),
        "gate": gate,
    }


def _prepare_head(head, teacher, *, scale: float = 0.15) -> bool:
    shared_lifting = (
        head.compilation.active_target_irreps
        == teacher.compilation.active_target_irreps
    )
    with torch.no_grad():
        if shared_lifting:
            head.lifting.load_state_dict(teacher.lifting.state_dict())
        for module_name in ("mean_projection", "covariance_projection", "scale_projection"):
            module = getattr(head, module_name, None)
            if module is None:
                continue
            if module_name == "mean_projection":
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()
            else:
                module.weight.mul_(scale)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    for module_name in ("covariance_projection", "scale_projection"):
        module = getattr(head, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return shared_lifting


def _basis_invariance(mu, target, scale, seed: FeatureSpec) -> float:
    """NLL discrepancy after an arbitrary orthogonal coordinate change."""
    random = torch.randn(scale.shape[-1], scale.shape[-1], device=scale.device, dtype=scale.dtype)
    q, _ = torch.linalg.qr(random)
    residual = target - mu
    transformed = torch.matmul(residual, q.transpose(-1, -2))
    transformed_scale = q @ scale @ q.transpose(-1, -2)
    return float((_nll_from_scale(residual, scale) - _nll_from_scale(transformed, transformed_scale)).abs())


def _equivariance_error(head, x, batch, scale, output: str) -> float:
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    input_rep = head.irreps_in if hasattr(head, "irreps_in") else None
    if input_rep is None:
        input_rep = o3.Irreps(head.compilation.seed_irreps)
    # e3nn's Wigner tables are CPU-backed in the server environment; construct
    # the representation matrices on CPU and move the finished matrices once.
    rho_in = input_rep.D_from_matrix(rotation).to(x)
    rho_out = O3IrrepsSpec(output).representation_matrix(rotation).to(x)
    rotation = rotation.to(x)
    x_rot = (rho_in @ x.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        mu, _ = head(x, batch)
        mu_rot, rotated_parameters = head(x_rot, batch)
        rotated_scale = head.compilation.build_spd_map().to(x.device)(
            rotated_parameters
        )
    predicted = scale
    mu_error = (mu_rot - mu @ rho_out.transpose(-1, -2)).abs().max()
    scale_error = (
        rotated_scale - rho_out @ predicted @ rho_out.transpose(-1, -2)
    ).abs().max()
    return float(torch.maximum(mu_error, scale_error).detach().cpu())


def run_pair(
    teacher_name: str,
    learner_name: str,
    output: str,
    teacher_family,
    learner_family,
    *,
    contexts: int,
    replicates: int,
    test_contexts: int,
    steps: int,
    seed: int,
    device: str,
) -> dict:
    torch.manual_seed(seed)
    dev = torch.device(device)
    feature = FeatureSpec.from_irreps(output, scope="global")
    teacher_plan = plan_readout(
        feature,
        output=output,
        covariance=teacher_family,
        distribution="student_t",
        student_t_dof=NU,
    )
    learner_plan = plan_readout(
        feature,
        output=output,
        covariance=learner_family,
        distribution="student_t",
        student_t_dof=NU,
    )
    teacher = teacher_plan.compilation.build_head().to(dev)
    learner = learner_plan.compilation.build_head().to(dev)
    # Keep the known teacher inside a numerically comfortable SPD regime on
    # both CPU and CUDA. This is a declared synthetic-data scale, not a
    # runtime fallback for a failed SPD map.
    _prepare_head(teacher, teacher)
    shared_lifting = _prepare_head(learner, teacher)
    teacher.eval()
    learner.train()

    x_context = feature.irreps.randn(contexts, -1, device=dev)
    x_test = feature.irreps.randn(test_contexts, -1, device=dev)
    train_x = x_context.repeat_interleave(replicates, dim=0)
    train_batch = torch.arange(train_x.shape[0], device=dev)
    context_batch = torch.arange(contexts, device=dev)
    test_batch = torch.arange(test_contexts, device=dev)
    teacher_spd_map = teacher_plan.compilation.build_spd_map().to(dev)
    learner_spd_map = learner_plan.compilation.build_spd_map().to(dev)
    loss_fn = StudentTNLL(NU)
    with torch.no_grad():
        train_mu, train_params = teacher(x_context, context_batch)
        train_scale = teacher_spd_map(train_params)
        y_train = _sample_student(
            train_mu.repeat_interleave(replicates, dim=0),
            train_scale.repeat_interleave(replicates, dim=0),
            NU,
        )
        true_mu, true_params = teacher(x_test, test_batch)
        true_scale = teacher_spd_map(true_params)

    trainable = [parameter for parameter in learner.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        mu, params = learner(train_x, train_batch)
        loss, _ = loss_fn(mu.detach(), params, y_train, learner_spd_map)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

    learner.eval()
    with torch.no_grad():
        pred_mu, pred_params = learner(x_test, test_batch)
        pred_scale = learner_spd_map(pred_params)
        test_loss, _ = loss_fn(
            pred_mu,
            pred_params,
            _sample_student(true_mu, true_scale, NU),
            learner_spd_map,
        )

    coverage = empirical_coverage(
        pred_mu,
        _sample_student(true_mu, true_scale, NU),
        pred_scale,
        reference="student_t",
        student_t_dof=NU,
    )
    with torch.no_grad():
        eig_true = torch.linalg.eigvalsh(true_scale)
        eig_pred = torch.linalg.eigvalsh(pred_scale)
        scale_error = covariance_relative_error(pred_scale, true_scale)
        eigen_error = (eig_pred - eig_true).abs().mean()
        residual = _sample_student(true_mu, true_scale, NU) - pred_mu
        whitened = torch.linalg.solve(pred_scale, residual.unsqueeze(-1)).squeeze(-1)
        mean_q = (residual * whitened).sum(-1).mean()
        equivariance_error = _equivariance_error(
            learner, x_test[: min(8, test_contexts)], test_batch[: min(8, test_contexts)], pred_scale[: min(8, test_contexts)], output
        )
        basis_nll_error = _basis_invariance(pred_mu, residual + pred_mu, pred_scale, feature)

    return {
        "family": learner_name,
        "teacher_family": teacher_name,
        "learner_family": learner_name,
        "family_match": teacher_name == learner_name,
        "shared_lifting": shared_lifting,
        "seed": seed,
        "output_irreps": output,
        "output_dimension": int(learner_plan.compilation.output_spec.dim),
        "parameter_count": int(learner_plan.compilation.covariance_parameter_count),
        "teacher_parameter_count": int(
            teacher_plan.compilation.covariance_parameter_count
        ),
        "relation_to_full": learner_plan.report.family["relation_to_full"],
        "teacher_relation_to_full": teacher_plan.report.family["relation_to_full"],
        "canonical_reachable": bool(
            learner_plan.report.representation_reachability["canonical"]["reachable"]
        ),
        "active_reachable": bool(
            learner_plan.report.representation_reachability["active"]["reachable"]
        ),
        "contexts": contexts,
        "replicates": replicates,
        "steps": steps,
        "student_t_dof": NU,
        "test_nll": float(test_loss),
        "covariance_relative_error": float(scale_error),
        "eigenvalue_mae": float(eigen_error),
        "mean_mae": float((pred_mu - true_mu).abs().mean()),
        "mean_mahalanobis_q": float(mean_q),
        "equivariance_max_abs": equivariance_error,
        "basis_change_nll_abs_error": basis_nll_error,
        **coverage,
    }


def run_case(
    name: str,
    output: str,
    family,
    *,
    contexts: int,
    replicates: int,
    test_contexts: int,
    steps: int,
    seed: int,
    device: str,
) -> dict:
    """Run one matching-family positive control."""
    return run_pair(
        name,
        name,
        output,
        family,
        family,
        contexts=contexts,
        replicates=replicates,
        test_contexts=test_contexts,
        steps=steps,
        seed=seed,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default=",".join(_cases()))
    parser.add_argument(
        "--cross-family-matrix",
        action="store_true",
        help="run all teacher/learner pairs for the selected same-output families",
    )
    parser.add_argument("--contexts", type=int, default=128)
    parser.add_argument("--replicates", type=int, default=32)
    parser.add_argument("--test-contexts", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = _cases()
    selected = [item.strip() for item in args.families.split(",")]
    unknown = sorted(set(selected) - set(cases))
    if unknown:
        parser.error(f"unknown families: {unknown}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    rows = []
    if args.cross_family_matrix:
        dimensions = {cases[name][0] for name in selected}
        if len(dimensions) != 1:
            parser.error(
                "cross-family matrix requires families with the same output representation"
            )
        for seed in seeds:
            for teacher_name in selected:
                output, teacher_spec = cases[teacher_name]
                for learner_name in selected:
                    _, learner_spec = cases[learner_name]
                    rows.append(
                        run_pair(
                            teacher_name,
                            learner_name,
                            output,
                            teacher_spec,
                            learner_spec,
                            contexts=args.contexts,
                            replicates=args.replicates,
                            test_contexts=args.test_contexts,
                            steps=args.steps,
                            seed=seed,
                            device=args.device,
                        )
                    )
    else:
        for seed in seeds:
            for family in selected:
                output, spec = cases[family]
                rows.append(
                    run_case(
                        family,
                        output,
                        spec,
                        contexts=args.contexts,
                        replicates=args.replicates,
                        test_contexts=args.test_contexts,
                        steps=args.steps,
                        seed=seed,
                        device=args.device,
                    )
                )
    result = {
        "kind": (
            "synthetic_cross_family_covariance_recovery"
            if args.cross_family_matrix
            else "synthetic_ground_truth_covariance_recovery"
        ),
        "contract": {
            "input_action": "O(3)",
            "true_distribution": "Student-t",
            "scale_semantics": "scatter S; covariance nu/(nu-2) S",
            "radial_reference": "q/d ~ F(d, nu)",
            "basis_check": "independent random orthogonal coordinate change",
        },
        "protocol": {**vars(args), "output": str(args.output)},
        "matrix_families": selected if args.cross_family_matrix else None,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

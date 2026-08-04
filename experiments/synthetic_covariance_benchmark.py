"""Controlled covariance-recovery benchmark for the statistical closure.

Each row uses a compiler-produced teacher and a learner with the same feature
contract.  The teacher generates repeated observations from a known Student-t
scale; the learner sees only the repeated observations and must recover the
declared full, low-rank, isotypic-block, or graph-precision family.  The
benchmark also checks group equivariance and invariance of the proper score
under an independent orthogonal coordinate change.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from scipy.stats import f

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
from compatibility.e3nn import o3


NU = 5.0


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


def _prepare_head(head, teacher, *, scale: float = 0.15) -> None:
    with torch.no_grad():
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
        dtype=x.dtype,
        device=x.device,
    )
    input_rep = head.irreps_in if hasattr(head, "irreps_in") else None
    if input_rep is None:
        input_rep = o3.Irreps(head.compilation.seed_irreps)
    rho_in = input_rep.D_from_matrix(rotation)
    rho_out = O3IrrepsSpec(output).representation_matrix(rotation).to(x)
    x_rot = (rho_in @ x.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        mu, _ = head(x, batch)
        mu_rot, _ = head(x_rot, batch)
    predicted = scale
    rotated_scale = head.compilation.build_spd_map()(head(x_rot, batch)[1])
    mu_error = (mu_rot - mu @ rho_out.transpose(-1, -2)).abs().max()
    scale_error = (
        rotated_scale - rho_out @ predicted @ rho_out.transpose(-1, -2)
    ).abs().max()
    return float(torch.maximum(mu_error, scale_error))


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
    torch.manual_seed(seed)
    dev = torch.device(device)
    feature = FeatureSpec.from_irreps(output, scope="global")
    plan = plan_readout(
        feature,
        output=output,
        covariance=family,
        distribution="student_t",
        student_t_dof=NU,
    )
    teacher = plan.compilation.build_head().to(dev)
    learner = plan.compilation.build_head().to(dev)
    _prepare_head(learner, teacher)
    teacher.eval()
    learner.train()

    x_context = feature.irreps.randn(contexts, -1, device=dev)
    x_test = feature.irreps.randn(test_contexts, -1, device=dev)
    train_x = x_context.repeat_interleave(replicates, dim=0)
    train_batch = torch.arange(train_x.shape[0], device=dev)
    context_batch = torch.arange(contexts, device=dev)
    test_batch = torch.arange(test_contexts, device=dev)
    spd_map = plan.compilation.build_spd_map().to(dev)
    loss_fn = StudentTNLL(NU)
    with torch.no_grad():
        train_mu, train_params = teacher(x_context, context_batch)
        train_scale = spd_map(train_params)
        y_train = _sample_student(
            train_mu.repeat_interleave(replicates, dim=0),
            train_scale.repeat_interleave(replicates, dim=0),
            NU,
        )
        true_mu, true_params = teacher(x_test, test_batch)
        true_scale = spd_map(true_params)

    trainable = [parameter for parameter in learner.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        mu, params = learner(train_x, train_batch)
        loss, _ = loss_fn(mu.detach(), params, y_train, spd_map)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

    learner.eval()
    with torch.no_grad():
        pred_mu, pred_params = learner(x_test, test_batch)
        pred_scale = spd_map(pred_params)
        test_loss, _ = loss_fn(pred_mu, pred_params, _sample_student(true_mu, true_scale, NU), spd_map)

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
        "family": name,
        "seed": seed,
        "output_irreps": output,
        "output_dimension": int(plan.compilation.output_spec.dim),
        "parameter_count": int(plan.compilation.covariance_parameter_count),
        "relation_to_full": plan.report.family["relation_to_full"],
        "canonical_reachable": bool(
            plan.report.representation_reachability["canonical"]["reachable"]
        ),
        "active_reachable": bool(
            plan.report.representation_reachability["active"]["reachable"]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default=",".join(_cases()))
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
        "kind": "synthetic_ground_truth_covariance_recovery",
        "contract": {
            "input_action": "O(3)",
            "true_distribution": "Student-t",
            "scale_semantics": "scatter S; covariance nu/(nu-2) S",
            "radial_reference": "q/d ~ F(d, nu)",
            "basis_check": "independent random orthogonal coordinate change",
        },
        "protocol": {**vars(args), "output": str(args.output)},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

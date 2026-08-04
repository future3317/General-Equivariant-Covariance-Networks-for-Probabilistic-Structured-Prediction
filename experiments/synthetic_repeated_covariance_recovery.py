"""Recover a known equivariant covariance with repeated observations.

This experiment separates implementation/optimization failures from the
single-label dielectric identification problem. A teacher compiler head
defines an equivariant scale map; each context is observed repeatedly under a
known Student-t law, while a learner sees only samples and the same features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from distributions import StudentTNLL
from equivcompiler import FeatureSpec, FullCovariance, plan_readout
from evaluation import covariance_relative_error


def _sample_student(mu: torch.Tensor, scale: torch.Tensor, nu: float) -> torch.Tensor:
    chol = torch.linalg.cholesky(scale)
    z = torch.randn_like(mu)
    chi = torch.distributions.Chi2(
        torch.tensor(nu, dtype=mu.dtype, device=mu.device)
    ).sample(mu.shape[:-1])
    return mu + torch.einsum("nij,nj->ni", chol, z) * torch.sqrt(nu / chi).unsqueeze(-1)


def _eigenbasis_offdiagonal_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    _, eigenvectors = torch.linalg.eigh(true)
    in_true_basis = eigenvectors.transpose(-1, -2) @ pred @ eigenvectors
    diagonal = torch.diag_embed(torch.diagonal(in_true_basis, dim1=-2, dim2=-1))
    off_diagonal = in_true_basis - diagonal
    return float(
        (
            torch.linalg.matrix_norm(off_diagonal)
            / torch.linalg.matrix_norm(in_true_basis).clamp_min(1e-12)
        ).mean()
    )


def run(
    *,
    contexts: int = 128,
    replicates: int = 32,
    steps: int = 1000,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    if contexts < 4 or replicates < 2:
        raise ValueError("use at least four contexts and two repeated observations")
    torch.manual_seed(seed)
    dev = torch.device(device)
    plan = plan_readout(
        FeatureSpec.from_irreps("0e+2e"),
        output="0e+2e",
        covariance=FullCovariance(),
        distribution="student_t",
        student_t_dof=5.0,
    )
    spd_map = plan.compilation.build_spd_map().to(dev)
    teacher = plan.compilation.build_head().to(dev)
    learner = plan.compilation.build_head().to(dev)
    with torch.no_grad():
        teacher.mean_projection.weight.zero_()
        learner.mean_projection.weight.zero_()
        teacher.covariance_projection.weight.mul_(0.05)
        learner.lifting.load_state_dict(teacher.lifting.state_dict())
        learner.covariance_projection.weight.zero_()
        if teacher.mean_projection.bias is not None:
            teacher.mean_projection.bias.zero_()
            learner.mean_projection.bias.zero_()
    for parameter in learner.mean_projection.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    contexts_x = torch.randn(contexts, 6, device=dev)
    test_x = torch.randn(max(32, contexts // 2), 6, device=dev)
    batch_context = torch.arange(contexts, device=dev)
    batch_test = torch.arange(test_x.shape[0], device=dev)
    with torch.no_grad():
        teacher_mu, teacher_params = teacher(contexts_x, batch_context)
        teacher_scale = spd_map(teacher_params)
        _test_mu, test_params = teacher(test_x, batch_test)
        test_scale = spd_map(test_params)
    x_train = contexts_x.repeat_interleave(replicates, dim=0)
    batch_train = torch.arange(x_train.shape[0], device=dev)
    with torch.no_grad():
        train_mu = teacher_mu.repeat_interleave(replicates, dim=0)
        train_scale = teacher_scale.repeat_interleave(replicates, dim=0)
        y_train = _sample_student(train_mu, train_scale, 5.0)
    trainable = [p for p in learner.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-4)
    loss_fn = StudentTNLL(5.0)
    learner.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        mu, params = learner(x_train, batch_train)
        loss, _ = loss_fn(mu.detach(), params, y_train, spd_map)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
    learner.eval()
    with torch.no_grad():
        _, pred_params = learner(test_x, batch_test)
        pred_scale = spd_map(pred_params)
    return {
        "contexts": contexts,
        "replicates": replicates,
        "steps": steps,
        "seed": seed,
        "distribution": "student_t",
        "covariance_relative_error": float(covariance_relative_error(pred_scale, test_scale)),
        "eigenbasis_offdiagonal_error": _eigenbasis_offdiagonal_error(pred_scale, test_scale),
        "true_log_eigenvalue_mae": float(
            (
                torch.log(torch.linalg.eigvalsh(test_scale))
                - torch.log(torch.linalg.eigvalsh(pred_scale))
            ).abs().mean()
        ),
        "teacher_test_scale_trace": float(
            torch.diagonal(test_scale, dim1=-2, dim2=-1).sum(-1).mean()
        ),
        "learner_test_scale_trace": float(
            torch.diagonal(pred_scale, dim1=-2, dim2=-1).sum(-1).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, default=128)
    parser.add_argument("--replicates", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = run(
        contexts=args.contexts,
        replicates=args.replicates,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

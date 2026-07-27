"""Controlled causal test for isospectral orientation calibration.

The base scale has fixed eigenvalues.  A teacher EIOC generates a known
equivariant orientation corruption; a student EIOC is then fitted by Gaussian
NLL.  The script records the NLL change and verifies that the student cannot
improve by changing eigenvalues, determinant, or condition number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from compatibility.e3nn import o3

from models import EquivariantIsospectralOrientationCalibrator


def gaussian_nll(scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    chol = torch.linalg.cholesky(scale)
    solved = torch.cholesky_solve(target.unsqueeze(-1), chol).squeeze(-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (logdet + (target * solved).sum(-1)).mean()


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    hidden = o3.Irreps("4x0e + 2x1o + 2x2e")
    output = o3.Irreps("0e + 2e")
    n, d = args.samples, output.dim
    features = torch.randn(n, hidden.dim, device=device)
    raw_scale = torch.diag(torch.tensor(
        [0.2, 0.35, 0.6, 0.9, 1.3, 1.8], dtype=torch.float32, device=device
    ))
    base_scale = raw_scale.expand(n, d, d).clone()

    teacher = EquivariantIsospectralOrientationCalibrator(hidden, output, zero_init=False).to(device)
    with torch.no_grad():
        for parameter in teacher.coefficient_head.parameters():
            parameter.normal_(std=args.teacher_std)
    true_scale = teacher(features, base_scale).detach()
    target = torch.bmm(
        torch.linalg.cholesky(true_scale),
        torch.randn(n, d, 1, device=device),
    ).squeeze(-1)

    student = EquivariantIsospectralOrientationCalibrator(hidden, output).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    initial_nll = float(gaussian_nll(base_scale, target).item())
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = student(features, base_scale)
        loss = gaussian_nll(prediction, target)
        loss.backward()
        optimizer.step()
    calibrated = student(features, base_scale).detach()
    final_nll = float(gaussian_nll(calibrated, target).item())
    base_eigenvalues = torch.linalg.eigvalsh(base_scale)
    calibrated_eigenvalues = torch.linalg.eigvalsh(calibrated)
    max_spectrum_error = float(
        (base_eigenvalues - calibrated_eigenvalues).abs().max().item()
    )
    base_logdet = torch.linalg.slogdet(base_scale)[1]
    calibrated_logdet = torch.linalg.slogdet(calibrated)[1]
    max_logdet_error = float((base_logdet - calibrated_logdet).abs().max().item())
    base_condition = base_eigenvalues[..., -1] / base_eigenvalues[..., 0]
    calibrated_condition = calibrated_eigenvalues[..., -1] / calibrated_eigenvalues[..., 0]
    max_condition_error = float((base_condition - calibrated_condition).abs().max().item())
    return {
        "device": str(device),
        "samples": n,
        "steps": args.steps,
        "initial_nll": initial_nll,
        "final_nll": final_nll,
        "nll_improvement": initial_nll - final_nll,
        "max_spectrum_error": max_spectrum_error,
        "max_logdet_error": max_logdet_error,
        "max_condition_error": max_condition_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--teacher_std", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Synthetic covariance-recovery experiment with an equivariant teacher.

A frozen equivariant teacher generates mean :math:`\\mu(x)` and symmetric
operator :math:`A(x)` from input features :math:`x` in a fixed O(3)
representation. The student model must recover the resulting covariance field
:math:`S(x)=\\exp(A(x))`. Train and test datasets share the same teacher; only
the input samples and observation noise differ.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from compatibility.e3nn import o3
from scipy.stats import chi2

from representations import O3IrrepsSpec
from spd_maps import MatrixExponentialMap
from distributions import GaussianNLL
from models import (
    EquivariantMeanHead,
    O3QuadraticSymmetricOperatorHead,
    StructuredProbabilisticPredictor,
)
from models.backbone import EquivariantActivation


# Input representation for the synthetic teacher.
# Includes scalars, vectors, and rank-2 tensors so the teacher can produce
# non-trivial output covariances.
DEFAULT_INPUT_IRREPS = "4x0e + 2x1o + 2x2e"


class SyntheticBackbone(nn.Module):
    """Fake backbone that embeds an input irrep vector to hidden_irreps.

    Uses a small equivariant MLP so the model has enough capacity to recover a
    non-trivial covariance field while preserving the O(3) group structure.
    """

    def __init__(self, input_irreps: o3.Irreps, hidden_irreps: o3.Irreps):
        super().__init__()
        self.input_irreps = o3.Irreps(input_irreps)
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.irreps_out = self.hidden_irreps

        hidden_mid = "32x0e + 16x1o + 16x2e"
        self.embed = o3.Linear(self.input_irreps, hidden_mid)
        self.act = EquivariantActivation(hidden_mid)
        self.proj = o3.Linear(self.act.irreps_out, self.hidden_irreps)

    def forward(self, data):
        x = self.embed(data.x)
        x = self.act(x)
        x = self.proj(x)
        batch = torch.arange(x.shape[0], device=x.device)
        return x, batch


class EquivariantQuadraticTeacherOperator(nn.Module):
    """Frozen quadratic equivariant operator teacher.

    Produces :math:`\\operatorname{Sym}^2(V)` coefficients through a linear
    branch plus an ``o3.TensorSquare`` branch. This allows the teacher to
    generate high-:math:`\\ell` covariance components (e.g. ``4e`` for
    ``V = 0e + 2e``) even when the input representation only contains
    :math:`\\ell \\le 2`.
    """

    def __init__(
        self,
        input_irreps: o3.Irreps,
        output_spec: O3IrrepsSpec,
        bottleneck_irreps: o3.Irreps = "16x0e + 8x1o + 8x2e",
        A_scale: float = 0.3,
    ):
        super().__init__()
        self.input_irreps = o3.Irreps(input_irreps)
        self.output_spec = output_spec
        self.operator_basis = output_spec.symmetric_square()
        self.bottleneck_irreps = o3.Irreps(bottleneck_irreps)

        self.pre = o3.Linear(self.input_irreps, self.bottleneck_irreps)
        self.linear = o3.Linear(
            self.bottleneck_irreps, self.operator_basis.operator_irreps
        )
        self.square = o3.TensorSquare(
            self.bottleneck_irreps,
            irreps_out=self.operator_basis.operator_irreps,
        )

        with torch.no_grad():
            self.linear.weight.mul_(A_scale)
            self.square.weight.mul_(A_scale)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre(x)
        coeff = self.linear(z) + self.square(z)
        A = self.operator_basis.assemble(coeff)
        return 0.5 * (A + A.transpose(-1, -2))


class EquivariantTeacher(nn.Module):
    """Frozen equivariant teacher that generates mean and log-covariance."""

    def __init__(
        self,
        input_irreps: o3.Irreps,
        output_spec: O3IrrepsSpec,
        A_scale: float = 0.3,
        mu_scale: float = 0.5,
        use_quadratic_operator: bool = True,
    ):
        super().__init__()
        self.input_irreps = o3.Irreps(input_irreps)
        self.output_spec = output_spec
        self.use_quadratic_operator = use_quadratic_operator

        self.mean_map = o3.Linear(self.input_irreps, output_spec.irreps)

        if use_quadratic_operator:
            self.operator = EquivariantQuadraticTeacherOperator(
                self.input_irreps, output_spec, A_scale=A_scale
            )
        else:
            self.operator_basis = output_spec.symmetric_square()
            self.operator_map = o3.Linear(
                self.input_irreps, self.operator_basis.operator_irreps
            )
            with torch.no_grad():
                self.operator_map.weight.mul_(A_scale)

        # Scale initial weights so eigenvalues stay moderate after matrix_exp.
        with torch.no_grad():
            self.mean_map.weight.mul_(mu_scale)

        # Freeze teacher.
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        mu = self.mean_map(x)
        if self.use_quadratic_operator:
            A = self.operator(x)
        else:
            A_coeff = self.operator_map(x)
            A = self.operator_basis.assemble(A_coeff)
            A = 0.5 * (A + A.transpose(-1, -2))
        S = torch.linalg.matrix_exp(A)
        return mu, A, S


class SyntheticDataset:
    """Generate synthetic data with a shared frozen equivariant teacher."""

    def __init__(
        self,
        output_irreps: str,
        num_samples: int,
        teacher: EquivariantTeacher,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.output_irreps = output_irreps
        self.output_spec = O3IrrepsSpec(output_irreps)
        self.num_samples = num_samples
        self.teacher = teacher
        self.input_irreps = teacher.input_irreps

        # Random input features in the input representation space.
        self.x = self.input_irreps.randn(num_samples, -1)

    def __len__(self):
        return self.num_samples

    def generate(self):
        mu, A, S = self.teacher(self.x)
        eps = torch.randn_like(mu)
        S_sqrt = torch.linalg.matrix_exp(0.5 * A)
        y = mu + (S_sqrt @ eps.unsqueeze(-1)).squeeze(-1)
        return self.x, y, mu, A, S


def _make_data_object(x: torch.Tensor):
    """Wrap feature vector in a minimal PyG-like Data object."""
    class Data:
        pass

    data = Data()
    data.x = x
    data.edge_index = torch.zeros(2, 0, dtype=torch.long, device=x.device)
    data.edge_sh = torch.zeros(0, 1, device=x.device)
    data.edge_rbf = torch.zeros(0, 1, device=x.device)
    data.edge_weights = torch.zeros(0, device=x.device)
    data.batch = torch.arange(x.shape[0], device=x.device)
    return data


def coverage(y: torch.Tensor, mu: torch.Tensor, S: torch.Tensor, alpha: float = 0.9) -> float:
    """Empirical coverage of the alpha-level confidence ellipsoid."""
    d = y.shape[-1]
    residual = y - mu
    L = torch.linalg.cholesky(S)
    z = torch.linalg.solve_triangular(L, residual.unsqueeze(-1), upper=False).squeeze(-1)
    q = (z * z).sum(dim=-1)
    threshold = chi2.ppf(alpha, df=float(d))
    return (q < threshold).float().mean().item()


def principal_subspace_angle(S1: torch.Tensor, S2: torch.Tensor, k: int = 2) -> float:
    """Principal angle between top-k eigenspaces of two SPD matrices."""
    _, V1 = torch.linalg.eigh(S1)
    _, V2 = torch.linalg.eigh(S2)
    U1 = V1[..., -k:]
    U2 = V2[..., -k:]
    M = torch.matmul(U1.transpose(-1, -2), U2)
    _, s, _ = torch.linalg.svd(M)
    cos = torch.clamp(s[..., :k], -1.0, 1.0)
    angles = torch.acos(cos)
    return angles.mean().item()


def evaluate(pred_mu, pred_A, y, true_mu, true_A, true_S):
    pred_S = torch.linalg.matrix_exp(0.5 * (pred_A + pred_A.transpose(-1, -2)))

    rel_cov = (torch.norm(pred_S - true_S, dim=(-2, -1)) / torch.norm(true_S, dim=(-2, -1))).mean().item()
    log_eucl = torch.norm(pred_A - true_A, dim=(-2, -1)).mean().item()

    true_eig = torch.linalg.eigvalsh(true_S)
    pred_eig = torch.linalg.eigvalsh(pred_S)
    eig_err = torch.mean(torch.abs(pred_eig - true_eig), dim=-1).mean().item()

    subspace_angle = principal_subspace_angle(pred_S, true_S, k=min(2, true_S.shape[-1]))

    residual = y - pred_mu
    L = torch.linalg.cholesky(pred_S)
    z = torch.linalg.solve_triangular(L, residual.unsqueeze(-1), upper=False).squeeze(-1)
    whitened_cov = (z.unsqueeze(-1) @ z.unsqueeze(-2)).mean(dim=0)

    cov_50 = coverage(y, pred_mu, pred_S, alpha=0.5)
    cov_90 = coverage(y, pred_mu, pred_S, alpha=0.9)
    cov_95 = coverage(y, pred_mu, pred_S, alpha=0.95)

    mu_err = torch.mean(torch.abs(pred_mu - true_mu)).item()

    return {
        "mu_mae": mu_err,
        "cov_rel_error": rel_cov,
        "log_euclidean_error": log_eucl,
        "eigenvalue_error": eig_err,
        "subspace_angle": subspace_angle,
        "whitened_cov_trace": torch.trace(whitened_cov).item(),
        "coverage_50": cov_50,
        "coverage_90": cov_90,
        "coverage_95": cov_95,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_irreps", default="0e + 2e", help="e.g. '1o', '0e + 2e', '0e + 2e + 2o'")
    parser.add_argument("--input_irreps", default=DEFAULT_INPUT_IRREPS)
    parser.add_argument("--use_quadratic_teacher", action="store_true", default=True)
    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--num_test", type=int, default=500)
    parser.add_argument("--hidden_irreps", default="32x0e + 16x1o + 8x2e")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_dir", default="results/synthetic_covariance_recovery")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    output_spec = O3IrrepsSpec(args.output_irreps)
    input_irreps = o3.Irreps(args.input_irreps)

    # Single frozen teacher shared by train and test.
    teacher = EquivariantTeacher(
        input_irreps, output_spec, use_quadratic_operator=args.use_quadratic_teacher
    ).to(args.device)

    backbone = SyntheticBackbone(input_irreps, args.hidden_irreps)
    mean_head = EquivariantMeanHead(backbone.hidden_irreps, output_spec.irreps, pool=True)
    cov_head = O3QuadraticSymmetricOperatorHead(backbone.hidden_irreps, output_spec, pool=True)

    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=MatrixExponentialMap(),
        distribution=GaussianNLL(),
    ).to(args.device)

    train_ds = SyntheticDataset(args.output_irreps, args.num_train, teacher, seed=args.seed)
    test_ds = SyntheticDataset(args.output_irreps, args.num_test, teacher, seed=args.seed + 1)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Output representation: {args.output_irreps} (dim={output_spec.dim})")
    print(f"Symmetric-operator dim: {output_spec.symmetric_square().operator_dim}")

    x_train, y_train, _, _, _ = train_ds.generate()
    x_train = x_train.to(args.device)
    y_train = y_train.to(args.device)

    model.train()
    for epoch in range(args.num_epochs):
        perm = torch.randperm(args.num_train)
        total_loss = torch.tensor(0.0, device=args.device)
        num_samples = 0
        for i in range(0, args.num_train, args.batch_size):
            idx = perm[i : i + args.batch_size]
            x = x_train[idx]
            y = y_train[idx]

            data = _make_data_object(x)
            result = model(data, target=y, return_scale=False)
            loss = result["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = y.shape[0]
            total_loss += loss.detach() * batch_size
            num_samples += batch_size

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}: train_loss={total_loss.item() / num_samples:.4f}")

    model.eval()
    with torch.inference_mode():
        x_test, y_test, true_mu, true_A, true_S = test_ds.generate()
        data_test = _make_data_object(x_test.to(args.device))
        result = model(data_test, target=y_test.to(args.device), return_scale=False)
        pred_mu = result["mu"].cpu()
        pred_A = result["params"].cpu()

    metrics = evaluate(pred_mu, pred_A, y_test, true_mu, true_A, true_S)
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump({**vars(args), **metrics}, f, indent=2)

    torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pt"))


if __name__ == "__main__":
    main()

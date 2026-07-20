"""End-to-end checkpoint benchmark for exact dense-projector lowering.

The script performs inference, loss/backward, and timing only.  It builds the
original spherical-CG compiler graph and an exactly lowered graph, loads the
same checkpoint into both with ``strict=True``, and compares all observable
probabilistic outputs and parameter gradients on real ModelNet40 batches.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from data.modelnet40_inertia_dataset import (
    default_modelnet40_cache_path,
    default_modelnet40_graph_cache_path,
    get_modelnet40_inertia_loaders,
)
from data.tensor_conversions import irreps_to_voigt
from models import EquivariantBackbone
from models.pooling import mean_pool
from representations import CompilerConfig, O3RepresentationCompiler
from scripts.benchmarking import environment_record, measure


def _build_model(args: argparse.Namespace, backend: str, device: torch.device):
    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        atom_feature_dim=49,
        num_basis=args.num_basis,
        atom_features="learnable",
        tp_backend=args.tp_backend,
        cueq_method=(args.cueq_method if args.tp_backend == "cueq" else "naive"),
    )
    compilation = O3RepresentationCompiler(
        "0e + 2e",
        CompilerConfig(
            covariance="full",
            output_scope="global",
            objective="gaussian",
            backend=backend,
        ),
    ).compile(backbone.irreps_out)
    return compilation.build_model(backbone).to(device), compilation


def _load_models(args: argparse.Namespace, device: torch.device):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a model state dictionary")
    spherical, spherical_compilation = _build_model(args, "spherical_cg", device)
    lowered, lowered_compilation = _build_model(args, "cartesian_stf", device)
    spherical.load_state_dict(state, strict=True)
    lowered.load_state_dict(state, strict=True)
    spherical.eval()
    lowered.eval()
    return spherical, lowered, spherical_compilation, lowered_compilation


def _load_test_loader(args: argparse.Namespace):
    cache_path = args.cache_path or default_modelnet40_cache_path()
    graph_cache_path = args.graph_cache_path
    if graph_cache_path is None:
        candidate = default_modelnet40_graph_cache_path(
            cache_path, args.num_points, args.num_neighbors
        )
        if candidate.is_file():
            graph_cache_path = candidate
    _, _, test_loader = get_modelnet40_inertia_loaders(
        cache_path=cache_path,
        target_type=args.target_type,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_neighbors=args.num_neighbors,
        max_radius=args.max_radius,
        num_basis=args.num_basis,
        lmax=args.lmax,
        graph_cache_path=graph_cache_path,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    return test_loader, Path(cache_path), graph_cache_path


def _per_sample_nll(model, result: dict, target: torch.Tensor) -> torch.Tensor:
    residual = target - result["mu"]
    logdet, quad = model.spd_map.statistics(result["params"], residual)
    d = residual.shape[-1]
    return 0.5 * d * math.log(2.0 * math.pi) + 0.5 * (logdet + quad)


def _from_shared_features(model, node_features, batch_index, target) -> dict:
    """Evaluate one mapped head after a single shared backbone execution."""
    mu, params = model._predict(node_features, batch_index)
    loss, components = model.distribution(mu, params, target, model.spd_map)
    return {
        "mu": mu,
        "params": params,
        "scale": model.spd_map(params),
        "loss": loss,
        "components": components,
    }


@torch.inference_mode()
def _paired_inference(
    spherical,
    lowered,
    test_loader,
    *,
    device: torch.device,
    max_batches: int | None,
    non_blocking: bool,
) -> dict:
    totals = {
        name: {
            "loss": 0.0,
            "physical_abs": 0.0,
            "irreps_abs": 0.0,
            "irreps_sq": 0.0,
            "spd": 0,
            "minimum_eigenvalue": float("inf"),
        }
        for name in ("spherical_cg", "dense_projector")
    }
    maximum = {
        key: 0.0
        for key in (
            "mu",
            "A",
            "A_relative_frobenius",
            "S",
            "S_relative_frobenius",
            "reference_S_max_abs",
            "per_sample_loss",
        )
    }
    samples = 0
    elements = 0
    batches = 0
    for batch_index, batch in enumerate(test_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(device, non_blocking=non_blocking)
        target = batch.y_irreps
        node_features, batch_index = spherical.backbone(batch)
        outputs = {
            "spherical_cg": _from_shared_features(
                spherical, node_features, batch_index, target
            ),
            "dense_projector": _from_shared_features(
                lowered, node_features, batch_index, target
            ),
        }
        reference = outputs["spherical_cg"]
        candidate = outputs["dense_projector"]
        maximum["mu"] = max(
            maximum["mu"], float((reference["mu"] - candidate["mu"]).abs().max())
        )
        maximum["A"] = max(
            maximum["A"],
            float((reference["params"] - candidate["params"]).abs().max()),
        )
        operator_difference = torch.linalg.matrix_norm(
            reference["params"] - candidate["params"], ord="fro", dim=(-2, -1)
        )
        operator_norm = torch.linalg.matrix_norm(
            reference["params"], ord="fro", dim=(-2, -1)
        ).clamp_min(torch.finfo(reference["params"].dtype).tiny)
        maximum["A_relative_frobenius"] = max(
            maximum["A_relative_frobenius"],
            float((operator_difference / operator_norm).max()),
        )
        maximum["S"] = max(
            maximum["S"],
            float((reference["scale"] - candidate["scale"]).abs().max()),
        )
        scale_difference = torch.linalg.matrix_norm(
            reference["scale"] - candidate["scale"], ord="fro", dim=(-2, -1)
        )
        scale_norm = torch.linalg.matrix_norm(
            reference["scale"], ord="fro", dim=(-2, -1)
        ).clamp_min(torch.finfo(reference["scale"].dtype).tiny)
        maximum["S_relative_frobenius"] = max(
            maximum["S_relative_frobenius"],
            float((scale_difference / scale_norm).max()),
        )
        maximum["reference_S_max_abs"] = max(
            maximum["reference_S_max_abs"], float(reference["scale"].abs().max())
        )
        reference_nll = _per_sample_nll(spherical, reference, target)
        candidate_nll = _per_sample_nll(lowered, candidate, target)
        maximum["per_sample_loss"] = max(
            maximum["per_sample_loss"],
            float((reference_nll - candidate_nll).abs().max()),
        )

        batch_size = target.shape[0]
        y_voigt_std = batch.y_voigt_std.reshape(batch_size, -1).to(device)
        y_voigt_mean = batch.y_voigt_mean.reshape(batch_size, -1).to(device)
        target_voigt = irreps_to_voigt(target) * y_voigt_std + y_voigt_mean
        for name, result in outputs.items():
            prediction_voigt = (
                irreps_to_voigt(result["mu"]) * y_voigt_std + y_voigt_mean
            )
            residual = result["mu"] - target
            eigenvalues = torch.linalg.eigvalsh(result["scale"])
            totals[name]["loss"] += float(result["loss"]) * batch_size
            totals[name]["physical_abs"] += float(
                (prediction_voigt - target_voigt).abs().sum()
            )
            totals[name]["irreps_abs"] += float(residual.abs().sum())
            totals[name]["irreps_sq"] += float(residual.square().sum())
            totals[name]["spd"] += int((eigenvalues[:, 0] > 0).sum())
            totals[name]["minimum_eigenvalue"] = min(
                totals[name]["minimum_eigenvalue"], float(eigenvalues.min())
            )
        samples += batch_size
        elements += target.numel()
        batches += 1

    metrics = {}
    physical_elements = samples * 6
    for name, total in totals.items():
        metrics[name] = {
            "gaussian_nll": total["loss"] / samples,
            "physical_mae": total["physical_abs"] / physical_elements,
            "irreps_mae": total["irreps_abs"] / elements,
            "irreps_rmse": math.sqrt(total["irreps_sq"] / elements),
            "spd_rate": total["spd"] / samples,
            "minimum_eigenvalue": total["minimum_eigenvalue"],
        }
    metric_differences = {
        key: abs(metrics["spherical_cg"][key] - metrics["dense_projector"][key])
        for key in metrics["spherical_cg"]
    }
    return {
        "samples": samples,
        "batches": batches,
        "metrics": metrics,
        "absolute_metric_differences": metric_differences,
        "maximum_per_sample_discrepancy": maximum,
    }


def _gradient_discrepancy(spherical, lowered, batch) -> dict:
    gradients = {}
    input_gradients = {}
    with torch.no_grad():
        shared_features, batch_index = spherical.backbone(batch)
    for name, model in (("spherical_cg", spherical), ("dense_projector", lowered)):
        model.zero_grad(set_to_none=True)
        features = shared_features.detach().clone().requires_grad_()
        result = _from_shared_features(
            model, features, batch_index, batch.y_irreps
        )
        result["loss"].backward()
        input_gradients[name] = features.grad.detach().clone()
        gradients[name] = {
            parameter_name: parameter.grad.detach().clone()
            for parameter_name, parameter in model.joint_head.named_parameters()
            if parameter.grad is not None
        }
    if gradients["spherical_cg"].keys() != gradients["dense_projector"].keys():
        raise RuntimeError("lowered checkpoint changed the parameter-coordinate set")
    per_parameter = {
        name: float(
            (gradients["spherical_cg"][name] - gradients["dense_projector"][name])
            .abs()
            .max()
        )
        for name in gradients["spherical_cg"]
    }
    worst_name = max(per_parameter, key=per_parameter.get)
    return {
        "maximum_absolute_error": per_parameter[worst_name],
        "worst_parameter": worst_name,
        "parameters_compared": len(per_parameter),
        "shared_backbone_feature_gradient_max_error": float(
            (input_gradients["spherical_cg"] - input_gradients["dense_projector"])
            .abs()
            .max()
        ),
    }


def _time_model(model, batch, args, device: torch.device) -> dict:
    latest = None

    def prepare_model() -> None:
        model.zero_grad(set_to_none=True)

    def full_forward() -> None:
        nonlocal latest
        latest = None
        latest = model(batch, target=batch.y_irreps, return_scale=True)

    def full_forward_backward() -> None:
        full_forward()
        latest["loss"].backward()

    full = {
        "forward": measure(
            full_forward,
            prepare=None,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "forward_backward": measure(
            full_forward_backward,
            prepare=prepare_model,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }

    with torch.no_grad():
        cached_node_features, cached_batch = model.backbone(batch)
    cached_node_features = cached_node_features.detach().requires_grad_()
    head_latest = None

    def prepare_readout() -> None:
        model.joint_head.zero_grad(set_to_none=True)
        cached_node_features.grad = None

    def readout_forward() -> None:
        nonlocal head_latest
        head_latest = None
        mu, params = model.joint_head(cached_node_features, cached_batch)
        loss, _ = model.distribution(mu, params, batch.y_irreps, model.spd_map)
        scale = model.spd_map(params)
        head_latest = (loss, scale)

    def readout_forward_backward() -> None:
        readout_forward()
        head_latest[0].backward()

    readout = {
        "forward": measure(
            readout_forward,
            prepare=None,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "forward_backward": measure(
            readout_forward_backward,
            prepare=prepare_readout,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }

    head = model.joint_head
    with torch.no_grad():
        pooled = mean_pool(cached_node_features, cached_batch)
        compiled = head.lifting(pooled)
        cached_mean = head.mean_projection(compiled)
    compiled = compiled.detach().requires_grad_()
    covariance_latest = None

    def prepare_covariance() -> None:
        head.covariance_projection.zero_grad(set_to_none=True)
        compiled.grad = None

    def covariance_forward() -> None:
        nonlocal covariance_latest
        covariance_latest = None
        coefficients = head.covariance_projection(compiled)
        params = head.operator_basis.assemble(coefficients)
        loss, _ = model.distribution(
            cached_mean, params, batch.y_irreps, model.spd_map
        )
        scale = model.spd_map(params)
        covariance_latest = (loss, scale)

    def covariance_forward_backward() -> None:
        covariance_forward()
        covariance_latest[0].backward()

    covariance_tail = {
        "forward": measure(
            covariance_forward,
            prepare=None,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "forward_backward": measure(
            covariance_forward_backward,
            prepare=prepare_covariance,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }
    for phase in ("forward", "forward_backward"):
        readout[phase]["share_of_end_to_end"] = (
            readout[phase]["median_ms"] / full[phase]["median_ms"]
        )
        covariance_tail[phase]["share_of_end_to_end"] = (
            covariance_tail[phase]["median_ms"] / full[phase]["median_ms"]
        )
    return {
        "end_to_end": full,
        "probabilistic_readout_cached_backbone": readout,
        "covariance_tail_cached_lifting": covariance_tail,
    }


def benchmark(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    test_loader, cache_path, graph_cache_path = _load_test_loader(args)
    spherical, lowered, spherical_compilation, lowered_compilation = _load_models(
        args, device
    )
    batch = next(iter(test_loader)).to(device, non_blocking=args.pin_memory)
    inference = _paired_inference(
        spherical,
        lowered,
        test_loader,
        device=device,
        max_batches=args.max_test_batches,
        non_blocking=args.pin_memory and device.type == "cuda",
    )
    gradients = _gradient_discrepancy(spherical, lowered, batch)
    timings = {
        "spherical_cg": _time_model(spherical, batch, args, device),
        "dense_projector": _time_model(lowered, batch, args, device),
    }
    for phase in ("forward", "forward_backward"):
        baseline = timings["spherical_cg"]["end_to_end"][phase]["median_ms"]
        optimized = timings["dense_projector"]["end_to_end"][phase]["median_ms"]
        timings["dense_projector"]["end_to_end"][phase]["speedup"] = (
            baseline / optimized
        )
    return {
        "kind": "mapped_checkpoint_end_to_end_benchmark",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": {
            "cache_path": str(cache_path),
            "graph_cache_path": str(graph_cache_path),
            "target_type": args.target_type,
        },
        "environment": environment_record(device),
        "protocol": {
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "model_mode": "eval with gradients enabled for timing",
            "checkpoint_loading": "strict=True into both executors",
            "training_steps": 0,
            "equivalence_input": (
                "one shared frozen backbone execution; executor timing reruns the "
                "complete model separately"
            ),
            "covariance_tail_definition": (
                "covariance projection, operator assembly, Gaussian NLL, and "
                "explicit scale after a cached shared lifting output"
            ),
        },
        "spherical_compilation": spherical_compilation.as_dict(),
        "lowered_compilation": lowered_compilation.as_dict(),
        "inference": inference,
        "gradient_equivalence": gradients,
        "timings": timings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--graph-cache-path", type=Path)
    parser.add_argument("--target-type", choices=("inertia", "shape_covariance"), default="inertia")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-basis", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-neighbors", type=int, default=16)
    parser.add_argument("--max-radius", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp-backend", choices=("e3nn", "cueq"), default="cueq")
    parser.add_argument("--cueq-method", choices=("naive", "fused_tp"), default="fused_tp")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--max-test-batches", type=int)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = benchmark(args)
    rendered = json.dumps(results, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

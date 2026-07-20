"""Profile the dielectric training loop to identify bottlenecks.

Reports per-component timings (data wait, H2D, backbone, heads, SPD/loss,
backward, optimizer), throughput, memory, and tensor-product statistics.
Saves a JSON summary and an optional Chrome trace.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from data.dielectric_dataset import get_dielectric_irreps_loaders
from data.paths import dataset_dir
from distributions import GaussianNLL
from models import (
    EquivariantBackbone,
    EquivariantMeanHead,
    O3QuadraticSymmetricOperatorHead,
    StructuredProbabilisticPredictor,
)
from representations import O3IrrepsSpec
from spd_maps import MatrixExponentialMap
from scripts._common import add_tensor_product_arguments, tensor_product_kwargs


def count_tp_instructions(model):
    """Return total weight_numel and instruction count across TP layers."""
    total_weight = 0
    instructions = []
    for name, module in model.named_modules():
        if hasattr(module, "weight_numel"):
            total_weight += module.weight_numel
            instructions.append(
                {
                    "name": name,
                    "weight_numel": module.weight_numel,
                    "instruction_count": len(module.instructions)
                    if hasattr(module, "instructions")
                    else None,
                    "irreps_in1": str(module.irreps_in1)
                    if hasattr(module, "irreps_in1")
                    else None,
                    "irreps_in2": str(module.irreps_in2)
                    if hasattr(module, "irreps_in2")
                    else None,
                    "irreps_out": str(module.irreps_out)
                    if hasattr(module, "irreps_out")
                    else None,
                }
            )
    return total_weight, instructions


def _to_device(batch, device, non_blocking):
    """Move a PyG batch to device, preserving non_blocking for pinned memory."""
    return batch.to(device, non_blocking=non_blocking)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_components(
    model, dataloader, device, warmup_batches: int, profile_batches: int
):
    """Synchronized component-level timing.

    Each batch is synchronized so that CPU-side timers correspond to GPU work.
    This is accurate for per-component attribution but does not reflect normal
    CPU/GPU overlap, so throughput numbers here are lower bounds.
    """
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    iterator = iter(dataloader)

    # Warmup.
    for _ in range(warmup_batches):
        batch = next(iterator)
        batch = _to_device(batch, device, non_blocking=False)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue
        result = model(batch, target=batch.y_irreps, return_scale=False)
        result["loss"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    data_wait_times = []
    h2d_times = []
    backbone_times = []
    mean_head_times = []
    cov_head_times = []
    loss_times = []
    backward_times = []
    clip_times = []
    optimizer_times = []

    total_graphs = 0
    total_nodes = 0
    total_edges = 0

    for _ in range(profile_batches):
        # Data wait: time to fetch the next batch from the dataloader.
        t0 = time.perf_counter()
        batch = next(iterator)
        t1 = time.perf_counter()
        data_wait_times.append(1000.0 * (t1 - t0))

        # H2D transfer.
        _sync(device)
        t0 = time.perf_counter()
        batch = _to_device(batch, device, non_blocking=False)
        _sync(device)
        t1 = time.perf_counter()
        h2d_times.append(1000.0 * (t1 - t0))

        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue

        # Backbone.
        _sync(device)
        t0 = time.perf_counter()
        node_features, graph_batch = model.backbone(batch)
        _sync(device)
        t1 = time.perf_counter()
        backbone_times.append(1000.0 * (t1 - t0))

        # Mean head.
        t0 = time.perf_counter()
        mu = model.mean_head(node_features, graph_batch)
        _sync(device)
        t1 = time.perf_counter()
        mean_head_times.append(1000.0 * (t1 - t0))

        # Covariance head.
        t0 = time.perf_counter()
        params = model.covariance_head(node_features, graph_batch)
        _sync(device)
        t1 = time.perf_counter()
        cov_head_times.append(1000.0 * (t1 - t0))

        # Loss (distribution computes logdet and precision action).
        t0 = time.perf_counter()
        loss, _ = model.distribution(mu, params, batch.y_irreps, model.spd_map)
        loss = loss.mean()
        _sync(device)
        t1 = time.perf_counter()
        loss_times.append(1000.0 * (t1 - t0))

        # Backward.
        t0 = time.perf_counter()
        loss.backward()
        _sync(device)
        t1 = time.perf_counter()
        backward_times.append(1000.0 * (t1 - t0))

        # Gradient clipping.
        t0 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        _sync(device)
        t1 = time.perf_counter()
        clip_times.append(1000.0 * (t1 - t0))

        # Optimizer step.
        t0 = time.perf_counter()
        optimizer.step()
        _sync(device)
        t1 = time.perf_counter()
        optimizer_times.append(1000.0 * (t1 - t0))

        optimizer.zero_grad(set_to_none=True)

        total_graphs += batch.y_irreps.shape[0]
        total_nodes += batch.node_features.shape[0]
        total_edges += batch.edge_index.shape[1]

    def mean(vals):
        return sum(vals) / max(len(vals), 1)

    return {
        "mode": "component_timing",
        "data_wait_ms": mean(data_wait_times),
        "h2d_ms": mean(h2d_times),
        "backbone_ms": mean(backbone_times),
        "mean_head_ms": mean(mean_head_times),
        "covariance_head_ms": mean(cov_head_times),
        "spd_loss_ms": mean(loss_times),
        "backward_ms": mean(backward_times),
        "grad_clip_ms": mean(clip_times),
        "optimizer_ms": mean(optimizer_times),
        "num_batches": len(data_wait_times),
        "total_graphs": total_graphs,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }


def profile_steady_state(
    model, dataloader, device, warmup_batches: int, profile_batches: int
):
    """Unsynchronized steady-state throughput measurement.

    Only synchronizes at the end, so this reflects normal CPU/GPU overlap and
    is the right number for graphs/s, nodes/s, edges/s.
    """
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    iterator = iter(dataloader)

    # Warmup.
    for _ in range(warmup_batches):
        batch = next(iterator)
        batch = _to_device(batch, device, non_blocking=False)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue
        result = model(batch, target=batch.y_irreps, return_scale=False)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    total_graphs = 0
    total_nodes = 0
    total_edges = 0

    _sync(device)
    start = time.perf_counter()
    for _ in range(profile_batches):
        batch = next(iterator)
        batch = _to_device(batch, device, non_blocking=False)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue
        result = model(batch, target=batch.y_irreps, return_scale=False)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_graphs += batch.y_irreps.shape[0]
        total_nodes += batch.node_features.shape[0]
        total_edges += batch.edge_index.shape[1]
    _sync(device)
    elapsed = time.perf_counter() - start

    return {
        "mode": "steady_state_throughput",
        "num_batches": profile_batches,
        "total_time_ms": elapsed * 1000.0,
        "total_graphs": total_graphs,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "graphs_per_second": total_graphs / elapsed,
        "nodes_per_second": total_nodes / elapsed,
        "edges_per_second": total_edges / elapsed,
    }


def profile_chrome_trace(
    model,
    dataloader,
    device,
    warmup_batches: int,
    active_batches: int,
    output_dir: Path,
):
    """Record a Chrome trace of complete training steps using torch.profiler."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    iterator = iter(dataloader)

    # Wait/warmup/active schedule. The trace captures the active phase only.
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        schedule=schedule(
            wait=warmup_batches, warmup=warmup_batches, active=active_batches
        ),
        on_trace_ready=tensorboard_trace_handler(str(output_dir)),
    )

    with prof:
        for step in range(2 * warmup_batches + active_batches):
            batch = next(iterator)
            batch = _to_device(batch, device, non_blocking=False)
            if batch.edge_index is None or batch.edge_index.numel() == 0:
                prof.step()
                continue

            optimizer.zero_grad(set_to_none=True)
            result = model(batch, target=batch.y_irreps, return_scale=False)
            loss = result["loss"].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            prof.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument(
        "--dataset_storage", choices=["files", "shards"], default="files"
    )
    parser.add_argument("--shard_cache_size", type=int, default=2)
    parser.add_argument("--save_dir", default="results/profile_dielectric")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_basis", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_batches", type=int, default=20)
    parser.add_argument("--profile_batches", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument(
        "--atom_features", default="manual", choices=["manual", "learnable"]
    )
    add_tensor_product_arguments(parser)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--chrome_trace", action="store_true")
    args = parser.parse_args()
    args.data_dir = str(dataset_dir(args.data_dir, "mp_dielectric"))

    device = torch.device(args.device)
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, _, _ = get_dielectric_irreps_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        lmax=args.lmax,
        storage=args.dataset_storage,
        shard_cache_size=args.shard_cache_size,
    )

    output_spec = O3IrrepsSpec("0e + 2e")
    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        atom_feature_dim=49,
        num_basis=args.num_basis,
        atom_features=args.atom_features,
        **tensor_product_kwargs(args),
    )
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3QuadraticSymmetricOperatorHead(
        backbone.irreps_out, output_spec, pool=True
    )
    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=MatrixExponentialMap(),
        distribution=GaussianNLL(),
    ).to(device)
    if args.compile_tp:
        model.backbone.compile_tensor_products(dynamic=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tp_weight_numel, tp_instructions = count_tp_instructions(model)

    print(f"Model parameters: {num_params:,}")
    print(f"TP total weight_numel: {tp_weight_numel:,}")

    # Reset peak memory before the measured region.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    component_timing = profile_components(
        model, train_loader, device, args.warmup_batches, args.profile_batches
    )
    steady_state = profile_steady_state(
        model, train_loader, device, args.warmup_batches, args.profile_batches
    )

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        peak_memory_mb = None

    summary = {
        "args": vars(args),
        "num_parameters": num_params,
        "tp_weight_numel": tp_weight_numel,
        "tp_instructions": tp_instructions,
        "component_timing_ms": component_timing,
        "steady_state": steady_state,
        "peak_cuda_memory_mb": peak_memory_mb,
    }

    with open(output_dir / "profile_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nComponent timing (ms/batch, synchronized):")
    for k in [
        "data_wait_ms",
        "h2d_ms",
        "backbone_ms",
        "mean_head_ms",
        "covariance_head_ms",
        "spd_loss_ms",
        "backward_ms",
        "grad_clip_ms",
        "optimizer_ms",
    ]:
        print(f"  {k}: {component_timing[k]:.3f}")

    print("\nSteady-state throughput:")
    print(f"  graphs/s: {steady_state['graphs_per_second']:.2f}")
    print(f"  nodes/s:  {steady_state['nodes_per_second']:.2f}")
    print(f"  edges/s:  {steady_state['edges_per_second']:.2f}")
    print(f"  total time: {steady_state['total_time_ms']:.1f} ms")

    if peak_memory_mb is not None:
        print(f"\nPeak CUDA memory: {peak_memory_mb:.2f} MB")

    if args.chrome_trace:
        profile_chrome_trace(
            model,
            train_loader,
            device,
            warmup_batches=args.warmup_batches,
            active_batches=args.profile_batches,
            output_dir=output_dir,
        )
        print(f"Chrome trace saved to {output_dir}")


if __name__ == "__main__":
    main()

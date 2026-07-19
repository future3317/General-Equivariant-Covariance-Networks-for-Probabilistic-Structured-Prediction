"""Profile the dielectric training loop to identify bottlenecks.

Reports per-component timings (data wait, H2D, backbone, heads, SPD/loss,
backward, optimizer), throughput, memory, and tensor-product statistics.
Saves a JSON summary and an optional Chrome trace.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.optim as optim

from data.dielectric_dataset import get_dielectric_irreps_loaders
from distributions import GaussianNLL
from models import (
    EquivariantBackbone,
    EquivariantMeanHead,
    O3QuadraticSymmetricOperatorHead,
    StructuredProbabilisticPredictor,
)
from representations import O3IrrepsSpec
from spd_maps import MatrixExponentialMap


def count_tp_instructions(model):
    """Return total weight_numel and instruction count across TP layers."""
    total_weight = 0
    instructions = []
    for name, module in model.named_modules():
        if hasattr(module, "weight_numel"):
            total_weight += module.weight_numel
            instructions.append({
                "name": name,
                "weight_numel": module.weight_numel,
                "irreps_in1": str(module.irreps_in1) if hasattr(module, "irreps_in1") else None,
                "irreps_in2": str(module.irreps_in2) if hasattr(module, "irreps_in2") else None,
                "irreps_out": str(module.irreps_out) if hasattr(module, "irreps_out") else None,
            })
    return total_weight, instructions


def profile_training_loop(model, dataloader, device, num_batches: int = 20):
    """Profile a short training loop and return timing breakdown."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    data_wait_times = []
    h2d_times = []
    backbone_times = []
    mean_head_times = []
    cov_head_times = []
    loss_times = []
    backward_times = []
    optimizer_times = []

    total_graphs = 0
    total_nodes = 0
    total_edges = 0

    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    def _record(start, end):
        if device.type == "cuda":
            return start.elapsed_time(end)  # ms
        return (end - start) * 1000.0  # s -> ms

    iter_start = time.time()
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        # Data wait.
        t0 = time.time() if device.type != "cuda" else torch.cuda.Event(enable_timing=True)
        if device.type == "cuda":
            t0.record()

        # Move to device.
        batch = batch.to(device, non_blocking=(dataloader.pin_memory if hasattr(dataloader, "pin_memory") else False))
        if device.type == "cuda":
            t1 = torch.cuda.Event(enable_timing=True)
            t1.record()
        else:
            t1 = time.time()

        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        # Backbone.
        if device.type == "cuda":
            t2 = torch.cuda.Event(enable_timing=True)
            t2.record()
        else:
            t2 = time.time()
        node_features, graph_batch = model.backbone(batch)
        if device.type == "cuda":
            t3 = torch.cuda.Event(enable_timing=True)
            t3.record()
        else:
            t3 = time.time()

        # Mean head.
        mu = model.mean_head(node_features, graph_batch)
        if device.type == "cuda":
            t4 = torch.cuda.Event(enable_timing=True)
            t4.record()
        else:
            t4 = time.time()

        # Covariance head.
        params = model.covariance_head(node_features, graph_batch)
        if device.type == "cuda":
            t5 = torch.cuda.Event(enable_timing=True)
            t5.record()
        else:
            t5 = time.time()

        # Loss (distribution computes logdet and precision action; no explicit scale).
        loss, _ = model.distribution(mu, params, batch.y_irreps, model.spd_map)
        loss = loss.mean()
        if device.type == "cuda":
            t6 = torch.cuda.Event(enable_timing=True)
            t6.record()
        else:
            t6 = time.time()

        # Backward.
        loss.backward()
        if device.type == "cuda":
            t7 = torch.cuda.Event(enable_timing=True)
            t7.record()
        else:
            t7 = time.time()

        # Optimizer step.
        optimizer.step()
        if device.type == "cuda":
            t8 = torch.cuda.Event(enable_timing=True)
            t8.record()
        else:
            t8 = time.time()

        # Synchronize for timing.
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Record intervals.
        # We cannot measure data wait precisely without a prefetch queue; use iteration gaps.
        data_wait_times.append(0.0)
        h2d_times.append(_record(t0, t1))
        backbone_times.append(_record(t1, t3))
        mean_head_times.append(_record(t3, t4))
        cov_head_times.append(_record(t4, t5))
        loss_times.append(_record(t5, t6))
        backward_times.append(_record(t6, t7))
        optimizer_times.append(_record(t7, t8))

        total_graphs += batch.y_irreps.shape[0]
        total_nodes += batch.node_features.shape[0]
        total_edges += batch.edge_index.shape[1]

    iter_end = time.time()
    total_time_ms = (iter_end - iter_start) * 1000.0

    # Approximate data wait from unaccounted time between batches.
    accounted_per_batch = [
        h + b + m + c + l + bw + o
        for h, b, m, c, l, bw, o in zip(
            h2d_times, backbone_times, mean_head_times, cov_head_times,
            loss_times, backward_times, optimizer_times,
        )
    ]
    # Simple heuristic: data wait is whatever is left of per-batch wall time.
    per_batch_wall = total_time_ms / len(accounted_per_batch) if accounted_per_batch else 0.0
    data_wait_times = [max(0.0, per_batch_wall - acc) for acc in accounted_per_batch]

    def mean(vals):
        return sum(vals) / max(len(vals), 1)

    return {
        "data_wait_ms": mean(data_wait_times),
        "h2d_ms": mean(h2d_times),
        "backbone_ms": mean(backbone_times),
        "mean_head_ms": mean(mean_head_times),
        "covariance_head_ms": mean(cov_head_times),
        "spd_loss_ms": mean(loss_times),
        "backward_ms": mean(backward_times),
        "optimizer_ms": mean(optimizer_times),
        "total_time_ms": total_time_ms,
        "num_batches": len(accounted_per_batch),
        "graphs_per_second": total_graphs / (total_time_ms / 1000.0),
        "nodes_per_second": total_nodes / (total_time_ms / 1000.0),
        "edges_per_second": total_edges / (total_time_ms / 1000.0),
        "total_graphs": total_graphs,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/mp_dielectric")
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_basis", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="results/profile_dielectric")
    parser.add_argument("--chrome_trace", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, _, _ = get_dielectric_irreps_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    output_spec = O3IrrepsSpec("0e + 2e")
    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        atom_feature_dim=49,
        num_basis=args.num_basis,
    )
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3QuadraticSymmetricOperatorHead(backbone.irreps_out, output_spec, pool=True)
    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=MatrixExponentialMap(),
        distribution=GaussianNLL(),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tp_weight_numel, tp_instructions = count_tp_instructions(model)

    print(f"Model parameters: {num_params:,}")
    print(f"TP total weight_numel: {tp_weight_numel:,}")

    timing = profile_training_loop(model, train_loader, device, num_batches=args.num_batches)

    if device.type == "cuda":
        timing["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        timing["peak_cuda_memory_mb"] = None

    summary = {
        "args": vars(args),
        "num_parameters": num_params,
        "tp_weight_numel": tp_weight_numel,
        "tp_instructions": tp_instructions,
        "timing": timing,
    }

    with open(output_dir / "profile_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTiming breakdown (ms/batch):")
    for k in [
        "data_wait_ms",
        "h2d_ms",
        "backbone_ms",
        "mean_head_ms",
        "covariance_head_ms",
        "spd_loss_ms",
        "backward_ms",
        "optimizer_ms",
    ]:
        print(f"  {k}: {timing[k]:.3f}")
    print(f"\nThroughput: {timing['graphs_per_second']:.2f} graphs/s, "
          f"{timing['nodes_per_second']:.2f} nodes/s, "
          f"{timing['edges_per_second']:.2f} edges/s")
    if timing["peak_cuda_memory_mb"] is not None:
        print(f"Peak CUDA memory: {timing['peak_cuda_memory_mb']:.2f} MB")

    if args.chrome_trace and device.type == "cuda":
        from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=tensorboard_trace_handler(str(output_dir)),
        ) as prof:
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= args.num_batches:
                    break
                batch = batch.to(device)
                if batch.edge_index is None or batch.edge_index.numel() == 0:
                    continue
                result = model(batch, target=batch.y_irreps, return_scale=False)
                result["loss"].backward()
        print(f"Chrome trace saved to {output_dir}")


if __name__ == "__main__":
    main()

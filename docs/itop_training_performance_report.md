# ITOP training performance report

This is a profiling record for the reviewer factorial run. It is not a
scientific result and does not justify changing the model or training
protocol.

## Baseline

- Commit: `3844f99` (clean server worktree)
- Command: `python -m scripts.run_itop_study --profile final --gpu 3 --skip_joint_finetune`
- Environment: `equivcompiler`, PyTorch 2.6.0+cu126, one RTX 4090, BF16 backbone
- Data: complete side-train cache, 17,991 frames; 512 points; 16 neighbors
- Training: batch 16, 8 persistent workers, 1,012 batches per epoch, seed 42
- Scientific contract: deterministic MSE followed by frozen uncertainty heads;
  same split, cache, optimizer, stopping rule, and evaluation for every arm

The first deterministic epoch took 653 s (1,012 batches, approximately 15.5
training samples/s). The second and third epoch validation records completed
without non-finite losses or gradients. A concurrent `nvidia-smi` snapshot
reported approximately 100% GPU utilization and 12.5 GB of 24.5 GB allocated
memory, so the observed bottleneck is GPU compute rather than a stalled input
pipeline.

## Decision

No performance patch was accepted in this factorial run. Increasing batch size,
changing the effective batch, enabling `torch.compile`, or changing the tensor
product backend would each introduce a separate runtime or optimization
variable and would invalidate the frozen comparison unless benchmarked with
the full numerical and scientific gates. The active run therefore remains on
the reference e3nn path.

The finite-precision SPD audit added in this revision is a correctness audit,
not a throughput optimization. It does not change the compiler family,
likelihood, precision islands, data order, or stopping policy.

## Reproduction

Run the command above from a clean worktree in the `equivcompiler` environment.
For a focused audit, use:

```text
python -m scripts.audit_spd_finite_precision \
  --output results/spd_finite_precision_audit.json
```

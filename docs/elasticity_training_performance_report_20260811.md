# Elasticity training pilot and performance gate

Date: 2026-08-11

Status: **Gate 1 passed; formal matched confirmation is authorized.** This is development evidence, not a manuscript result.

## Scientific and measurement contract

- source: clean server worktree at `914e48a3`;
- hardware: NVIDIA RTX 4090, one process per GPU;
- data: official MP elasticity files, deterministic seed-42 subset of 1,024 train and 256 validation/test samples;
- output: 21-dimensional rank-4 elasticity representation;
- arms: deterministic mean, rank-2 Low-rank Student-t, Full Student-t;
- probability: fixed `nu=5`, existing compiler/SPD map/Student-t NLL, FP32 training algebra and FP64 scatter audit;
- optimizer: AdamW, batch 16, identical LR/schedule/clipping, validation-only selection;
- measured region: all training and per-epoch validation, including DataLoader and transfers, excluding compiler construction and final test diagnostics;
- acceptance: finite/schema/SPD/artifact gates, validation improvement, and less than 20 GiB peak allocated memory.

## Pilot result

| Arm | First val criterion | Best val criterion | Test MAE (GPa) | Test NLL | Cov90 | Peak allocated | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic | 27.497 | 26.820 | 22.187 | n/a | n/a | 1.01 GiB | 333 s |
| Low-rank-t | 40.976 | 39.246 | 26.131 | 33.783 | 0.508 | 0.99 GiB | 334 s |
| Full-t | 40.756 | 32.902 | 29.626 | 30.487 | 0.430 | 1.31 GiB | 339 s |

All three arms completed six epochs and selected epoch 6. Full-t and Low-rank-t passed strict FP64 scatter reconstruction. Both rejected radius-direction independence at the minimum permutation resolution (`p=0.005`). The pilot therefore establishes trainability but also warns that the high-order predictive ellipses remain statistically misspecified.

The compiler schema records 231 Full operator coordinates and a canonical covariance representation reaching `l=8`; rank-2 Low-rank uses 43 active coordinates with the same canonical reference. This is the high-order execution path the formal experiment is intended to validate.

## Performance decision

A matched one-epoch Full-t probe used the same 256 train and 64 validation/test samples, architecture, seed, workers, optimizer, and GPU. Batch 16 achieved 14.03 examples/s with 0.71 GiB peak allocation; batch 64 achieved 9.87 examples/s with 1.87 GiB. The larger batch was **rejected**: it was 29.6% slower and used 2.62x peak allocated memory, while also changing the optimization batch contract.

No kernel, precision, or mathematical fast path was accepted. Formal runs retain eager FP32, batch 16, four persistent workers, pinned transfer, and prefetch factor 2. The only wall-clock optimization is running independent arm/seed jobs concurrently on otherwise idle GPUs; this does not change per-run numerical or scientific semantics.

## Formal gate

Proceed with seeds 42, 43, and 44 on the complete official splits, a shared cap of 12 epochs and patience 3, and the same three arms. Do not increase the batch, enable mixed precision, compile ragged end-to-end execution, or create a second probability implementation. Manuscript integration remains blocked until all nine cells pass artifact and metric audits.


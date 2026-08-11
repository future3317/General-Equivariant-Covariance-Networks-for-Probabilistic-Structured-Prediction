# High-Order Elasticity Evidence Design

## Goal

Turn the existing rank-4 elasticity compile-only case into a controlled end-to-end training result that directly supports the compiler's high-order structured-output claim, while deriving the remaining low-cost ITOP statistics from existing artifacts and avoiding redundant dielectric experiments.

## Audit decision

| Requested evidence | Current state | Decision |
|---|---|---|
| Elasticity Full-t / compact-t / mean training | Trainer partially supports compiled Full/Low-rank and Student-t; only a one-epoch Block smoke artifact exists; no credible evaluation or multi-seed result | Implement and run through staged gates |
| Dielectric global temperature | Implemented in `evaluation/temperature.py` and `scripts/calibrate_dielectric_temperature.py`; existing E0/E1 evidence already separates radial and directional failure | Do not rerun |
| Dielectric learned/conditional nu | Implemented and experimentally verified; improves NLL/radial KS but leaves direction defect and independence rejection unchanged | Complete diagnostic; do not rerun |
| Dielectric isotypic recalibration | No direct arm, but the 24-arm factorial already includes an isotypic Block family and rejects all single ellipses | Reject as redundant calibration scope |
| ITOP Graph-t versus Full-t paired Top NLL CI | Full and Graph per-sample predictions exist; no direct paired bootstrap report | Add artifact-only analysis |
| ITOP coordinate/memory Pareto | Coordinates and runtime/memory artifacts exist; evidence is scattered | Consolidate without training |
| Five ITOP seeds | Existing three runs are frozen-head seeds sharing one backbone, not independent initializations | Do not expand; report the contract accurately |
| External non-equivariant baseline | Deterministic and independent/diagonal probabilistic baselines already exist; a new Cholesky baseline would add scope without addressing the high-order gap | Defer |

## Scientific isolation

The elasticity comparison changes only the predictive family. Full Student-t, Low-rank Student-t, and deterministic mean arms use identical train/validation/test membership, target normalization, graph construction, backbone architecture, optimizer, schedule, early stopping, and seed policy. Probabilistic arms use the public typed compiler and existing Student-t likelihood/SPD maps. The deterministic arm uses the existing equivariant backbone and `DeterministicHead`; it is a point-prediction control, not a compiler operator-family claim.

The output type is the 21-dimensional rank-4 elasticity representation. The compiled Full and Low-rank schemas must record canonical/active reachability, operator coordinates, retained instructions, exact lowering mode, and highest reachable coupling degree. This is the primary evidence that high-order compilation is exercised during real training.

## Staged training protocol

### Gate 0: harness and finite-contract smoke

Run one batch on CPU or one free GPU for deterministic, Low-rank-t, and Full-t. Require valid schemas, finite forward/loss/backward, SPD certificate for probabilistic arms, and reproducible split hashes.

### Gate 1: fast one-seed pilot

Use seed 42, 1,024 training examples, 256 validation/test examples, 256-atom/graph representation unchanged, at most 6 epochs, and patience 2. Require:

- all required artifacts and hashes;
- finite predictions and FP64 scatter reconstruction;
- validation criterion improves from its first finite epoch;
- no arm exceeds 20 GiB peak allocated GPU memory;
- measured examples/s and peak memory are recorded;
- Full-t and Low-rank-t complete evaluation with exact normalized NLL, Energy Score, Coverage90/95, radial KS, direction defect, and radius-direction independence.

Failure stops expansion and triggers harness/performance diagnosis. Scientific ranking is not inferred from the pilot.

### Gate 2: formal matched confirmation

Only after Gate 1, run seeds 42, 43, and 44 on the full official training split, with the same architecture and controls, at most 30 epochs and patience 5. Early stopping is validation-only. A run is publishable evidence when all three arms complete for all three seeds with finite predictions, reproducible provenance, and no family-specific data or optimization changes. The result need not beat point-prediction SOTA; it must demonstrate stable trainability and evaluability.

## Evaluation and artifacts

Every arm writes `args.json`, `environment.json`, `schema.json`, `history.json`, `metrics.json`, `predictions.pt`, `best_model.pt`, and `train.log`. Metrics include mean MAE/RMSE, exact normalized NLL for probabilistic arms, Energy Score, Coverage90/95, calibration error, E0-style elliptical diagnostics, active operator coordinates, selected epoch, trainable parameters, examples/s, wall time, and peak allocated/reserved GPU memory. Prediction tensors include sample IDs, mean, target, and compiler parameters; scatter is reconstructed in FP64 for audit rather than duplicated in the artifact.

The study manifest records source state, data-file hashes, split hashes, seed, command, compiler schema, checkpoint hash, prediction hash, and finite checks. These are repository evidence and will not be copied into manuscript prose.

## Performance contract

System optimization must preserve output type, probability law, SPD/operator algebra, FP32 loss path, data membership/order policy, effective batch, optimizer, schedule, and selection rule. Profile the fast pilot first. Accept only measured changes that improve median examples/s by at least 5% without failing numerical, equivariance, gradient, checkpoint, or metric gates. Likely safe candidates are deterministic dataset subset selection, precomputed graph caching, pinned-memory transfer, and removing repeated CPU graph construction; mixed precision is restricted to the backbone and is not enabled without a matched numerical gate.

## ITOP artifact-only analysis

Use the existing paired Full-t and Graph-t Side/Top prediction artifacts. Verify identical target/sample ordering and frozen-backbone provenance. Bootstrap samples, not aggregate seed means, to estimate the paired Graph-minus-Full Top NLL difference and 95% percentile interval. Report coordinates and measured peak memory/latency from existing audited runtime records. The result is a frozen-backbone family comparison and must not be described as independent-backbone robustness.

## Stop rules and manuscript gate

- Do not rerun dielectric temperature, conditional-nu, family factorial, mixture, conformal, or spectral-window experiments.
- Do not add ITOP seeds or train external baselines in this phase.
- Do not update the manuscript until the formal elasticity gate and ITOP artifact audit both pass.
- If Full-t is computationally impractical after one measured optimization pass, retain Low-rank-t as trained high-order evidence and report Full as a measured compile/inference resource boundary; do not silently reduce the output type or change the statistical law.


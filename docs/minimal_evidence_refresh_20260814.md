# Minimal evidence refresh

Date: 2026-08-14  
Status: evidence-backed consolidation; no new training was required.

## Decision

The reviewer-recommended minimal evidence design is already covered by
complete artifacts. No new model, seed, hyperparameter, or compiler path was
started in this refresh.

| Evidence module | Primary comparison | Formal evidence | Secondary evidence |
|---|---|---|---|
| Dielectric family/law separation | Isotropic vs. Full x Gaussian vs. fixed-nu Student-t, seeds 42--44 | `RESULTS/Tpami/dielectric/family_factorial_formal_0b5ea92/factorial_result.json` | Isotypic-block and low-rank rows in the same factorial |
| High-order reachability | Deterministic + Full-t, seeds 42--44 | `RESULTS/Tpami/Elasticity/formal_feb75b9/study_manifest.json` | Low-rank-t active-target reference |
| Structured output under view shift | Full-t vs. true Graph-t vs. degree-matched shuffled Graph-t | `RESULTS/Tpami/ITOP/reviewer_controls_2c7cb38/control_audit.json` plus the three-seed family artifacts | Low-rank, no-edge, and fixed-coordinate one-seed diagnostics |

The formal runs use the server's `equivcompiler` environment and the declared
validation-only selection rules. For ITOP, Side is used for training and
validation; Top is evaluation-only. No Top/OOD value is used for stopping,
tuning, or model selection.

## What the artifacts support

- Dielectric: the four headline cells are sufficient to separate operator
  family from radial-law choice. The existing Full Student-t result is the
  strongest fixed-law arm; the conditional-nu confirmation separately improves
  radial/tail diagnostics without removing directional misspecification.
- Elasticity: the legacy-Voigt deterministic and Full-t runs establish
  end-to-end training of the complete 231-coordinate, l=8 path. The
  representation-compatible normalization follow-up is a feasibility
  diagnostic: seed 42 is finite, while seed 43 fails fast on a non-finite loss;
  it is not a corrected-normalization headline result.
- ITOP: the exact seed-42 true-vs.-shuffled topology comparison changes only
  the declared graph topology at matched 174 coordinates, and the additional
  shuffled seeds confirm initialization robustness. The result is a topology-
  specific likelihood effect, not a claim of calibrated Top uncertainty.

## Performance audit decision

The existing profiling record is `results/profiles/comparison_bs32.csv` and
`docs/itop_training_performance_report.md`. The measured final protocol is GPU
compute-bound (approximately 100% GPU utilization in the recorded window); a
representative 20-batch window reports about 1.03 s data wait versus 28.47 s
total steady-state time. Changing batch size, workers, precision, tensor-
product backend, or compilation would change the optimization/runtime
contract and would require a new matched scientific study. No performance
patch is therefore accepted in this evidence refresh.

## Verification record

- Local `egnn` focused regression tests: 67 passed.
- Server `equivcompiler` focused regression tests: 65 passed.
- Local Ruff and server Ruff: passed.
- Local full `pytest`: aborted in the existing ModelNet40 inertia dataset
  construction test; this is retained as a verification limitation and is not
  reclassified as a pass.
- Server formal training artifacts remain unchanged. No new TPAMI process was
  launched during this refresh.
- The Graph-t structured diagnostics figure, including panel (a), is not
  modified.

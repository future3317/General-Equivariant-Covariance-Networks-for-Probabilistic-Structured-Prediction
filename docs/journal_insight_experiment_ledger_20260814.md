# Journal insight experiment ledger

This ledger is the source of truth for the long-horizon evidence refresh. It
distinguishes existing artifacts from new runs and keeps test evaluation out of
selection decisions.

## Baseline artifacts already present

| Evidence | Artifact | Protocol status |
|---|---|---|
| Dielectric Full-t factorial | `results/dielectric_family_factorial_0b5ea92/full/student_t/seed_{42,43,44}` | Existing formal checkpoints; validation-only selection; 281 test structures |
| Dielectric paired bootstrap | `results/dielectric_paired_factorial_bootstrap_20260814.json` | Read-only audit of existing predictions; no test selection |
| ITOP true Graph-t robustness | `results/itop_graph_t_robustness_ec25e58/seed_{42,43,44}/frozen_graph_student_t` | Existing three-seed Graph-t artifacts; exact pairing is certified by `results/itop_topology_pairing_audit_20260817_subject_cluster.json` |
| ITOP shuffled controls | `results/itop_reviewer_controls_2c7cb38/seed_{42,43,44}/shuffled_graph_student_t` plus `results/itop_reviewer_controls_matched_20260816/seed_{43,44}` | Completed three-seed true-vs-shuffled topology pairing; shared protocol/cache, effective split seed, frame IDs, and targets verified per seed; pooled paired $\Delta$NLL is +20.987 (Side) and +26.820 (Top), with positive subject-level effects for all four subject clusters; the pooled contrast is distinct from seed-averaged marginal means and is reported descriptively; official Zenodo IDs encode person/frame only (`XX_YYYYY`), with no action-sequence field |
| Elasticity formal study | `results/elasticity_end_to_end_feb75b9` | Existing legacy-Voigt deterministic/LR-t/Full-t study; not representation-compatible evidence |
| Elasticity compatible feasibility | `results/elasticity_stability_20260816/{D2_seed43,D3_seed43,D4_seed43}` plus server-side checkpoints | D2 and D3 fail fast under unrestricted Full; D4 bounded-window diagnostic is finite and strict-SPD for seed 43; no promoted unrestricted result |
| External Deep Ensemble control | `reviewer_external_controls_20260818/dielectric_ensemble_formal_20260818`, seeds 42, 43, and 44; staged mean → covariance → joint training; `ensemble_3member_metrics.json`; artifact source commit `5141d709903fe36c63b168f64f86a6981d9d0d60` | Three independently initialized members from the compiled O(3) dielectric path share one inference contract. The exact density is an equally weighted Student-t member mixture; raw test mixture NLL is $-3.595$ and Energy Score $0.405$, versus validation-calibrated NLL $-3.190$ and Energy Score $0.419$. Coverage is the explicitly named moment-Gaussian diagnostic, so this remains an external diagnostic and not a fixed-coordinate/non-equivariant or headline benchmark comparison |

## Current decision ledger

| Stage | Hypothesis | Required evidence | Status | Next action |
|---|---|---|---|---|
| Matched dielectric conditional-nu | Conditional radial law improves the headline Full-t checkpoints, not only a separate frozen checkpoint | Same seed, features, mean, scatter, 281 test IDs; validation-only conditional-nu fit; paired law-correct metrics | Completed positive diagnostic | Keep fixed-$\nu$ vs. conditional-$\nu(x)$ as the formal law-adaptation comparison; retain directional failure as a limitation |
| Exact ITOP topology | True skeleton helps beyond a split or initialization artifact | True/shuffled Graph-t paired within seeds 42/43/44, with shared protocol/cache, effective split seed, frame IDs, and targets | Completed | Report the paired likelihood effect; retain no-edge and fixed-coordinate as one-seed diagnostics |
| Random topology distribution | True skeleton is better than most degree-matched random trees | Pre-generated trees, no outcome filtering, fixed split and protocol | Pending | Generate topology manifest before training |
| Compatible elasticity | Failure is a repairable numerical boundary rather than incompatible target semantics | FP32/FP64 replay, failing-term trace, shifted-log oracle, multiplicity whitening, and one bounded-window control | Completed negative diagnostic | Stop expansion; retain bounded-window result only as a limitation/control |
| Law-aware diagnostics | Diagnostic reference belongs to the registered predictive law | Contract tests for Gaussian, Student-t, conditional-nu, and mixture | Partial implementation | Consolidate existing methods and remove experiment-name dispatch |
| K=2 mixture | A two-scatter mixture explains residual non-elliptical structure | Exact mixture NLL plus law-correct joint diagnostic improvement | Prototype exists; promotion pending | Audit existing mixture path, then run matched dielectric pilot |
| Group backend | Group-independent IR has a small backend registration boundary | Static dependency audit plus minimal O(2)/SO(2) oracle if feasible | Pending | Perform architecture audit after core evidence |

## Non-negotiable selection rules

- Train/validation may select checkpoints and hyperparameters; test is never used
  for selection.
- ITOP Top is evaluation-only.
- Any new repair remains exploratory until its method, hyperparameters, and
  stopping rule are frozen from development/validation evidence and evaluated
  once on a separately hash-locked confirmation split.
- A mixture cannot be promoted from NLL alone; it needs a law-correct joint
  diagnostic improvement.
- A representation-compatible elasticity claim requires all predetermined seeds
  to be finite and pass strict-SPD/equivariance checks.
- Existing fixed-law results and compiler semantics are not overwritten.

## Accepted systems-only optimization

The E1 runner now accepts `--diagnostic_splits`. The default remains
`train val test`, preserving the historical behavior; formal law-comparison
runs use `val test` so that train predictions and train NLL remain available
while the expensive train Energy/elliptical diagnostics are explicitly marked
`skipped_by_config`. This changes no model, loss, split, seed, checkpoint
selection rule, or prediction artifact. A server smoke run with the matched
operator projection produced complete predictions, protocol, diagnostics,
environment, and manifest artifacts under
`journal_insight_20260814/dielectric/smoke_seed42_retry3`.

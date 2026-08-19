# Journal insight experiment ledger

This ledger is the source of truth for the long-horizon evidence refresh. It
distinguishes existing artifacts from new runs and keeps test evaluation out of
selection decisions.

## Baseline artifacts already present

| Evidence | Artifact | Protocol status |
|---|---|---|
| Dielectric Full-t factorial | `results/dielectric_family_factorial_0b5ea92/full/student_t/seed_{42,43,44}` | Existing formal checkpoints; validation-only selection; 281 test structures |
| Dielectric paired bootstrap | `results/dielectric_paired_factorial_bootstrap_20260814.json` | Read-only audit of existing predictions; no test selection |
| ITOP true Graph-t robustness | `results/itop_graph_t_robustness_ec25e58/seed_{42,43,44}/frozen_graph_student_t` | Existing three-seed Graph-t artifacts; exact pairing is certified by `results/itop_topology_pairing_audit_20260819_corrected.json` |
| ITOP shuffled controls | `results/itop_reviewer_controls_2c7cb38/seed_{42,43,44}/shuffled_graph_student_t` plus `results/itop_reviewer_controls_matched_20260816/seed_{43,44}` | Completed three-seed true-vs-shuffled topology pairing; shared protocol/cache, effective split seed, frame IDs, and targets verified per seed; corrected pooled paired $\Delta$NLL is +20.944 (Side) and +28.039 (Top), with positive subject-level effects for all four subject clusters; the same FP64 Student-$t$ sufficient-statistic estimand is used for paired and marginal means; official Zenodo IDs encode person/frame only (`XX_YYYYY`), with no action-sequence field |
| Elasticity formal study | `results/elasticity_end_to_end_feb75b9` | Existing legacy-Voigt deterministic/LR-t/Full-t study; not representation-compatible evidence |
| Elasticity compatible feasibility | `results/elasticity_stability_20260816/{D2_seed43,D3_seed43,D4_seed43}` plus server-side checkpoints | D2 and D3 fail fast under unrestricted Full; D4 bounded-window diagnostic is finite and strict-SPD for seed 43; no promoted unrestricted result |
| External Deep Ensemble control | `reviewer_external_controls_20260818/dielectric_ensemble_formal_20260818`, seeds 42, 43, and 44; staged mean → covariance → joint training; `ensemble_3member_metrics.json`; artifact source commit `5141d709903fe36c63b168f64f86a6981d9d0d60` | Three independently initialized members from the compiled O(3) dielectric path share one inference contract. The exact density is an equally weighted Student-t member mixture; raw test mixture NLL is $-3.595$ and Energy Score $0.405$, versus validation-calibrated NLL $-3.190$ and Energy Score $0.419$. Coverage is the explicitly named moment-Gaussian diagnostic, so this remains an external diagnostic and not a fixed-coordinate/non-equivariant or headline benchmark comparison |
| Compiler-integrated orientation refinement | `results/orientation_pilot_20260819_final` plus the server-side completion log | Artifact-only completion is complete and finite, but the frozen conditional-$\nu$ base remains structurally misspecified: test NLL is $-2.622$ (worse than the matched conditional-$\nu$ reference), radius--direction dependence remains $0.466$ with permutation $p=0.005$, and spherical-direction diagnostics reject at $\alpha=0.01$. The predeclared one-pilot stop rule therefore rejects seed expansion and does not promote orientation or mixture claims. |
| Shared-mean K=2 mixture pilot | `results/shared_mean_mixture_pilot_20260819` plus server-side predictions/checkpoint | One validation-only pilot after the orientation stop rule. Exact finite-mixture test NLL is $-2.825$ and Energy Score $0.442$; all predictions are finite and FP64 component-scatter SPD passes (minimum eigenvalue $1.81\times10^{-4}$ on validation and $2.35\times10^{-4}$ on test), but mixture-aware projection PIT still rejects with 48/64 Bonferroni rejections and pooled decile $L_1=0.289$. It does not improve the matched conditional-$\nu$ reference and is not promoted or expanded. |
| Representation-compatible elasticity Full-t | `results/elasticity_stability_20260819/asinh_exp_formal_20260819/seed_{42,43,44}` plus compact local JSON/artifacts | The asinh full-image spectral chart is a semantics-compatible candidate distinct from the legacy Voigt stress test. All three seeds have complete artifacts and pass finite/FP64-SPD/active-$\ell=8$/validation-only gates. Mean NLL/Energy/MAE are $18.809\pm0.389/2.788\pm0.023/13.333\pm0.713$; the result is reported separately and not cross-protocol ranked against legacy Voigt. |

## Current decision ledger

| Stage | Hypothesis | Required evidence | Status | Next action |
|---|---|---|---|---|
| Matched dielectric conditional-nu | Conditional radial law improves the headline Full-t checkpoints, not only a separate frozen checkpoint | Same seed, features, mean, scatter, 281 test IDs; validation-only conditional-nu fit; paired law-correct metrics | Completed positive diagnostic | Keep fixed-$\nu$ vs. conditional-$\nu(x)$ as the formal law-adaptation comparison; retain directional failure as a limitation |
| Exact ITOP topology | True skeleton helps beyond a split or initialization artifact | True/shuffled Graph-t paired within seeds 42/43/44, with shared protocol/cache, effective split seed, frame IDs, and targets | Completed | Report the paired likelihood effect; retain no-edge and fixed-coordinate as one-seed diagnostics |
| Random topology distribution | True skeleton is better than most degree-matched random trees | Pre-generated trees, no outcome filtering, fixed split and protocol | In progress | Finish and audit topology index 0 before launching the remaining manifest entries |
| Compatible elasticity | Failure is a repairable numerical boundary rather than incompatible target semantics | FP32/FP64 replay, failing-term trace, shifted-log oracle, multiplicity whitening, and one bounded-window control | Completed negative diagnostic | Stop expansion; retain bounded-window result only as a limitation/control |
| Orientation refinement | Remaining conditional-$\nu$ defect is explained by an isospectral orientation field | Frozen-base $\Lambda^2(V)$ compiler target, finite/SPD checks, NLL and direction diagnostics | Completed negative pilot | Do not expand seeds or start a mixture from this result; retain as a negative diagnostic only |
| Law-aware diagnostics | Diagnostic reference belongs to the registered predictive law | Contract tests for Gaussian, Student-t, conditional-nu, and mixture | Partial implementation | Consolidate existing methods and remove experiment-name dispatch |
| K=2 mixture | A two-scatter mixture explains residual non-elliptical structure | Exact mixture NLL plus law-correct joint diagnostic improvement | Completed negative pilot | Do not expand seeds, expose a public mixture law, or add a manuscript claim; retain the finite-mixture implementation as an evaluated diagnostic utility |
| Group backend | Group-independent IR has a small backend registration boundary | Static dependency audit plus minimal O(2)/SO(2) oracle if feasible | Pending | Perform architecture audit after core evidence |
| Asinh full-image elasticity | A milder surjective spectral chart can make the representation-compatible Full path numerically trainable | Three fixed-seed runs with the same target normalization, validation-only selection, finite predictions, FP64 strict-SPD, and equivariance | Completed eligible evidence | Report as a separate Full-chart appendix result; do not pool with legacy Voigt values |

## Non-negotiable selection rules

- Train/validation may select checkpoints and hyperparameters; test is never used
  for selection.
- ITOP Top is evaluation-only.
- Any new repair remains exploratory until its method, hyperparameters, and
  stopping rule are frozen from development/validation evidence and evaluated
  once on a separately hash-locked confirmation split.
- The orientation and shared-mean mixture pilots were diagnostic interventions
  run against existing frozen dielectric evaluation artifacts. Because they
  were not preceded by a new confirmation split, they are non-confirmatory and
  cannot support a positive manuscript claim.
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

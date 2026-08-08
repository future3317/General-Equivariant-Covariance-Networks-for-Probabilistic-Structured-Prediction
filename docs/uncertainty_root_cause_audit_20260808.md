# Uncertainty root-cause repository and artifact audit

Date: 2026-08-08

Scope: repository source, tests, local downloaded artifacts, server artifacts under
`/home/workspace/lrh/RESULTS/Tpami`, and the current manuscript. This audit is a
phase gate before adding uncertainty-model code or launching new training.

Evidence labels:

- **Implemented**: executable source exists and is connected to a callable API.
- **Pipeline**: reachable from a released trainer/evaluator, not only a helper.
- **Verified**: a non-smoke experiment has logs/checkpoints/predictions or metrics
  with an identifiable protocol and provenance.
- **Missing**: no equivalent source or no credible experiment artifact was found.

## Evidence matrix

| Requirement | Implemented / pipeline | Experimentally verified | Status | Evidence path |
|---|---|---|---|---|
| Fixed-ν multivariate Student-t NLL and scatter semantics | Yes / yes | Yes, synthetic, dielectric, and ITOP | Verified | `distributions/student_t.py`; `equivcompiler/distributions.py`; server dielectric unified run; local ITOP final runs |
| Conditional ν(x) from invariant features | No | No | Missing | Repository-wide search found only scalar `student_t_dof`; all 64 server run records carrying this field use 5.0 |
| Globally fitted ν | No optimizer/profile-likelihood path | No | Missing | No fitted-ν source, config, log, or artifact |
| K=2 equivariant Student-t mixture used as one predictor | No | No | Missing | No mixture distribution spec, head, loss, trainer model kind, checkpoint, or prediction artifact |
| Exact finite-ensemble mixture NLL | Yes / evaluator only | Unit-tested; no deep-ensemble result | Implemented, not experimentally verified | `evaluation/ensemble.py`; `scripts/evaluate_dielectric_ensemble.py`; `tests/test_ensemble.py` |
| Dielectric end-to-end deep ensemble | Evaluator exists; member-training orchestration absent | No | Missing experiment | No ensemble/mixture artifact on server; dielectric runs are single-seed except OOF folds, which are not ensemble members |
| ITOP end-to-end ensemble | Deterministic evaluator exists | No | Missing experiment | `scripts/evaluate_itop_ensemble.py`; server has only seed-42 full deterministic checkpoints; no `deterministic_ensemble/metrics.json` |
| ITOP frozen-head three-seed audit | Yes | Yes | Verified, but not E3 | `results/itop_family_robustness_75b2ee1`; `results/itop_graph_t_robustness_ec25e58`; all reuse backbone SHA `85e46d...` |
| Student-t radial PIT | Yes / dielectric audit | Yes for final unified dielectric checkpoint | Verified partially for E0 | `evaluation/metrics.py::student_t_radial_pit`; server unified joint `symmetry_audit.json` |
| Symmetric whitening and second-moment defect | Yes / dielectric audit | Yes | Verified partially for E0 | `symmetric_whitened_residuals`, `whitened_second_moment_defect`, and `irrep_resolved_whitening_defect`; unified joint audit |
| Random-direction projection PIT | Yes, added after the audit / E0 runner | No released cross-family artifact yet | Implemented, verification pending | `evaluation/elliptical.py`; `scripts/audit_elliptical_law.py` |
| Whitened-direction sphericality | Yes, added after the audit / E0 runner | No released cross-family artifact yet | Implemented, verification pending | Pure directional test in `evaluation/elliptical.py`; the legacy `whitened_angular_defect` remains only a compatibility alias for the joint defect |
| Radius-direction dependence | Yes, added after the audit / E0 runner | No released cross-family artifact yet | Implemented, verification pending | Signed/axial Spearman max-statistic permutation test in `evaluation/elliptical.py` |
| E0 stratification by ITOP visibility / dielectric descriptors | Yes, added after the audit / E0 runner | Partial descriptive ITOP evidence only; formal E0 artifacts pending | Implemented, verification pending | Semantic ITOP visibility bins and dielectric descriptor strata in `scripts/audit_elliptical_law.py` |
| Dielectric fixed-ν frozen-family baseline | Yes | Yes | Verified | Unified three-stage Student-t artifact and current manuscript tables |
| Strict matched spectral-window control | Required maps and covariance-only stage exist | No | Missing E1 control | Existing three saved runs differ in width/backend/precision/metric settings and are correctly labeled descriptive only |
| Unbounded matrix-exp control on identical frozen H, μ | Map exists and trainer accepts it | No matching artifact | Missing E1 control | `spd_maps/matrix_exp.py`; `scripts/train_dielectric.py`; no server dielectric `args.json` uses `matrix_exp` |
| ITOP frozen-H sufficiency probe | Frozen pooled features exist | No probe result | Missing E2 | Server `reviewer_factorial_3844f99/.../frozen_features/{side_train,side_test,top_test}.pt` |
| ITOP raw-observation descriptor probe | Raw/cache geometry exists | No probe source/result | Missing E2 | Server ITOP geometry caches contain points, centroids, neighbors, visibility; no matched probe runner/artifact |
| ITOP observation perturbation pushforward covariance | No | No | Missing E2 | No dropout/depth-noise/partial-visibility perturbation implementation or artifact |
| Side→Top observation-shift audit | Yes | Yes | Verified context, not sufficiency proof | `scripts/diagnose_itop_failure.py`; visibility/chamfer and final figure artifacts |
| E3 bootstrap/subsample member training | No general member protocol | No | Missing E3 | No bootstrap field in dielectric run arguments; no end-to-end ITOP member set |
| Repeated dielectric labels | N/A | No | Unavailable evidence | Dataset audit contains one fixed-protocol label per structure; OOF folds do not create repeated labels |
| Synthetic matching-family recovery | Yes | Yes, three seeds with repeated observations | Verified positive control | `results/closure_230e2df`; `results/closure_cross_9514995`; corresponding experiment runners |
| OOF faithful dielectric residual experiment | Yes | Yes, negative result | Verified No-Go; do not repeat | Server `dielectric/faithful_oof_ablation_20260728/{ordinary_faithful,oof_faithful}` |

Targeted regression tests executed during this audit and the minimal E0 addition:
`tests/test_distributions.py`, `tests/test_ensemble.py`,
`tests/test_mathematical_contract.py`, and
`tests/test_itop_training_control.py`, plus
`tests/test_elliptical_diagnostics.py`: **33 passed**.

The E0 implementation was added only after the baseline audit established that
the diagnostics were absent. It reuses the existing symmetric-whitening
primitive and SciPy reference distributions, adds no new distribution or SPD
parameterization, and has not yet been promoted to experimental evidence. A
preliminary Graph-t JSON was generated during development, but must be rerun
with the final semantic visibility strata and compared with Full-t before it is
used for a root-cause conclusion.

## Already done that overlaps the proposed work

1. Fixed-ν Student-t, correct scale/covariance conversion, Student-t marginal
   quantiles, and radial `q/d ~ F(d,ν)` semantics are implemented and tested.
2. Dielectric already has radial-PIT summaries, coordinate-independent symmetric
   whitening, second-moment and irrep-resolved defects, full 21-coordinate rank
   audit, Energy Score, sliced CRPS, coverage, and risk-coverage.
3. Exact finite-mixture NLL and total-covariance decomposition already exist as
   an external evaluator. The missing part is trained independent members, not
   another mixture-NLL implementation.
4. ITOP already has complete finite predictions for 4,863 Side and 4,863 Top
   samples, visibility, Mahalanobis statistics, frozen pooled feature caches,
   view-shift diagnostics, and head-seed robustness.
5. Matrix exponential, ordinary spectral window, and centered spectral window
   already share the typed operator/compiler path. A matched control needs
   orchestration and artifact discipline, not a new SPD primitive.

## Minimum missing experiments

### E0 — elliptical-law falsification

- **Hypothesis:** a single fixed-ν elliptical Student-t is already contradicted
  by whitened residual geometry.
- **Existing evidence:** dielectric radial and second-moment defects are bad;
  ITOP Side/Top coverage is bad, but neither is a complete falsification.
- **Intervention:** add only the missing diagnostics around existing symmetric
  whitening: projection PIT, direction sphericality, radius-direction
  dependence, and declared strata. Reuse existing predictions/checkpoints.
- **Controls:** fixed checkpoint, split, coordinates, scale semantics, ν, and
  deterministic random-direction seed.
- **Decisive outcome:** reject the single ellipse if calibrated radial behavior
  coexists with non-uniform projections/directions, or if direction statistics
  depend materially on radius/visibility/descriptors.
- **Interpretation:** separates radial-tail failure from directional/mixture
  misspecification without training a new model.

### E1 — frozen-family test plus matched spectral control

- **Hypothesis:** failure is radial heterogeneity, multimodality, or an SPD-window
  restriction.
- **Existing evidence:** only fixed ν is trained; existing window runs are not
  factorial.
- **Intervention:** on one frozen H and μ, compare fixed ν, invariant conditional
  ν, and K=2 equivariant Student-t mixture; for dielectric, compare matrix-exp,
  centered `exp(4)`, and a wider centered window with every other field fixed.
- **Controls:** checkpoint hashes, split, seed, FP32 operator algebra, optimizer,
  early stopping, and validation-only selection.
- **Decisive outcome:** conditional ν wins only if tail heterogeneity dominates;
  mixture wins if single-component topology is wrong; matrix-exp/wider window
  wins only if boundary restriction is causal.
- **Interpretation:** determines whether any statistical model change is needed
  before touching the backbone.

### E2 — ITOP information sufficiency

- **Hypothesis:** pooled mean-oriented H omits observation ambiguity information.
- **Existing evidence:** Top visibility and point-cloud geometry shift strongly;
  learned scatter does not track the error increase.
- **Intervention:** matched small probes for squared error from frozen H versus
  inference-time raw geometry descriptors, then perturbation-induced mean
  covariance from physically plausible point dropout/depth noise.
- **Controls:** identical diagnostic train/holdout IDs, probe capacity, target,
  regularization, and no benchmark/test-informed model selection.
- **Decisive outcome:** raw descriptors or perturbation covariance materially
  outperform H-derived predictions on held-out diagnostic data.
- **Interpretation:** identifies an observation/backbone bottleneck rather than
  a covariance-family failure.

### E3 — true end-to-end ensemble

- **Hypothesis:** model/function uncertainty missing from a frozen single
  backbone is a major source of residual error.
- **Existing evidence:** no end-to-end ensemble; existing ITOP three-seed audit
  varies only uncertainty-head training.
- **Intervention:** only if E0–E2 leave this hypothesis live, train 3 independent
  end-to-end members (bootstrap/subsample declared) and score the exact finite
  mixture.
- **Controls:** same architecture/protocol, independent initialization and data
  resampling, member-level provenance, no moment-matched pseudo-density.
- **Decisive outcome:** mixture NLL/coverage and between-member spread improve
  held-out prediction in a stable, source-interpretable way.
- **Interpretation:** evidence for model-predictive uncertainty, not physical
  aleatoric covariance.

## Reusable modules

- Distribution/compiler boundary: `equivcompiler/distributions.py`,
  `distributions/base.py`, `equivcompiler/modules.py`.
- Student-t density and semantics: `distributions/student_t.py`.
- Exact mixture evaluation: `evaluation/ensemble.py` and
  `scripts/evaluate_dielectric_ensemble.py`.
- Whitening/calibration primitives: `evaluation/metrics.py` and
  `evaluation/calibration.py`.
- Typed SPD maps: `spd_maps/matrix_exp.py`, `spd_maps/spectral_window.py`, and
  `spd_maps/centered_spectral_window.py`.
- Frozen-head controls: dielectric `configure_training_stage`; ITOP
  `ControlledMeanOperatorHead`, feature-cache loader, and reproducibility
  contracts.
- ITOP raw geometry and visibility: `data/itop_dataset.py`, geometry caches, and
  saved prediction artifacts.

## Manuscript consistency audit

Consistent with artifacts:

- The manuscript explicitly says dielectric has one fixed-protocol label per
  structure and does not identify physical aleatoric covariance.
- It labels the ensemble as an external, unreported wrapper and uses finite
  mixture NLL semantics correctly.
- It states that ITOP seeds 42/43/44 vary the frozen uncertainty head, not the
  backbone.
- It labels the existing dielectric spectral-window comparison as unmatched
  sensitivity evidence rather than a causal ablation.
- The reported unified dielectric values (90% coverage 0.7117, mean q 25.3224,
  second-moment defect 10.4612, and irrep defects 3.8130/5.4676/8.0625) trace to
  `/home/workspace/lrh/RESULTS/Tpami/dielectric/unified_student_t_centered_b128_20260726/joint`.

Corrections/risks:

1. The manuscript's OOF paragraph still says "angular whitening defect" even
   though the released metric is a joint second-moment defect and explicitly
   not a pure angular statistic. This wording should be changed when results
   are next updated.
2. `results/dielectric_figures_final_ef680c2` is a different legacy single-run
   artifact (roughly 0.669 coverage at 90% and q≈30.18), despite its ambiguous
   name. It must not be used to overwrite unified-run manuscript values or
   figures.
3. Paper figure files do not carry an adjacent machine-readable provenance
   manifest. At least the uncertainty-alignment PDF hash matches the unified
   server figure exactly; future regeneration should copy a manifest with
   checkpoint/run-spec hashes.

## Recommended execution order

1. Complete E0 from current artifacts; no training.
2. Use E0 to decide whether E1 needs conditional ν, K=2 mixture, or both;
   execute the matched spectral control in parallel only because it removes a
   known confounder using existing maps/stages.
3. Run E2 before any end-to-end retraining.
4. Run E3 only if E0–E2 leave model/function uncertainty as a live root cause.
5. Add a compiler-level mixture primitive only after E1 demonstrates that a
   mixture materially changes proper-score and diagnostic behavior.

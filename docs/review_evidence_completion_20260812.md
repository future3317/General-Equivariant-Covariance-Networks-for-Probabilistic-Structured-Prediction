# Review-evidence completion audit

This report records only artifact-backed additions made after the reviewer
check. It does not promote a new model or change the frozen main protocol.

## Existing evidence

| Item | Status | Evidence |
|---|---|---|
| Independent NumPy/SciPy teacher recovery | Complete | `results/independent_teacher_recovery_bc0d297/result.json`; 12/12 matched rows pass and the formal gate permits the dielectric factorial |
| Dielectric family × radial-law factorial | Complete | `results/dielectric_family_factorial_0b5ea92/factorial_result_with_coverage.json`; 24 matched arms, three seeds |
| Validation-fitted dielectric temperature control | Complete negative control | `results/dielectric_temperature_calibration_20260726/temperature_calibration.json`; global/block temperature worsens held-out test NLL |
| Elasticity end-to-end training | Complete | `results/elasticity_end_to_end_feb75b9/study_manifest.json`; deterministic, LR-t, and Full-t over seeds 42/43/44 |
| ITOP six uncertainty heads | Complete for one-seed factorial | `results/itop_reviewer_factorial_3844f99` plus seed-42 Full-t artifact in `results/itop_reviewer_final_full_f9e02f3` |
| ITOP Full/LR/Graph robustness | Complete for three seeds | `results/itop_family_robustness_75b2ee1`, `results/itop_graph_t_robustness_ec25e58`, and seed-42 Full/Graph artifacts |

## New evidence

### ITOP six-head single-seed audit

The six-head rows share the same seed-42 backbone checkpoint hash and matched
Side/Top targets and frame IDs. The table is a descriptive factorial audit,
not a three-seed claim.

| Head | Active coordinates | Side NLL | Top NLL | Side Energy | Top Energy | Side Cov90 | Top Cov90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-t | 1,035 | -70.891 | 9.633 | 0.724 | 2.893 | 0.832 | 0.159 |
| Indep-G | 90 | -19.932 | 754.155 | 0.727 | 2.453 | 0.725 | 0.116 |
| Indep-t | 90 | -27.050 | 41.130 | 0.728 | 2.442 | 0.776 | 0.168 |
| LR-t | 181 | -35.989 | 31.653 | 0.724 | 2.638 | 0.869 | 0.039 |
| Graph-G | 174 | -50.554 | 797.773 | 0.726 | 2.656 | 0.688 | 0.037 |
| Graph-t | 174 | -55.931 | 2.482 | 0.726 | 2.478 | 0.803 | 0.144 |

The coverage values are mean per-joint marginal coverages at the nominal
level. The exact source rows remain available in the audit JSON and original
metrics artifacts.

### ITOP paired per-frame bootstrap

The new audit uses exact saved frame IDs, identical targets, and production
Student-t sufficient statistics. Values are `right - left`; negative Top NLL
means the right-hand model is better under the proper score.

| Contrast | View | Pooled NLL difference | 95% paired bootstrap interval |
|---|---|---:|---:|
| Graph-t − Full-t | Side | +14.848 | [+14.691, +15.002] |
| Graph-t − Full-t | Top | -4.372 | [-4.492, -4.257] |
| Graph-t − LR-t | Side | -19.944 | [-20.102, -19.783] |
| Graph-t − LR-t | Top | -27.257 | [-27.362, -27.150] |
| LR-t − Full-t | Side | +34.793 | [+34.588, +34.994] |
| LR-t − Full-t | Top | +22.885 | [+22.766, +22.998] |

All three frozen families use the same deterministic mean within each seed,
so paired MPJPE differences are exactly zero. The bootstrap output is in
`results/itop_review_evidence_20260812/itop_review_evidence.json`.

### Dielectric paired factorial contrasts

The factorial predictions are also paired by the same 281 test structures and
the same frozen representation. The audit resamples structures as clusters,
averages repeated initialization seeds within each sampled structure, and
reports right-minus-left NLL:

| Contrast | Mean difference | 95% cluster bootstrap interval |
|---|---:|---:|
| Full Student-t - Full Gaussian | -11.313 | [-11.669, -10.966] |
| Full Student-t - Isotropic Student-t | -2.568 | [-2.936, -2.199] |

The machine-readable output is
`results/dielectric_paired_factorial_bootstrap_20260814.json`. This is an
artifact-backed uncertainty interval for the existing factorial; it does not
add training or test-set model selection.

### Dielectric fixed-nu sensitivity

This is a post-hoc scan of the saved Full-t predictions using the existing
Student-t log-probability implementation. It is not validation selection and
does not constitute a learned global nu experiment.

| Fixed nu | Test NLL mean ± seed SD |
|---:|---:|
| 3 | -2.573 ± 0.225 |
| 5 | -2.316 ± 0.245 |
| 10 | -1.697 ± 0.285 |
| 30 | -0.083 ± 0.394 |

The `nu=5` value reproduces the factorial Full-t row. The post-hoc grid is
slightly better at `nu=3`, so the manuscript must describe `nu=5` results as a
fixed-law factorial rather than as evidence that 5 is selected or universal.
The exact output is `results/dielectric_fixed_nu_sensitivity_20260812/fixed_nu_sensitivity.json`.

## Supported inference

- The ITOP graph advantage is a reproducible cross-view proper-NLL trade-off:
  Graph-t is worse than Full-t on matched Side frames but better on matched Top
  frames, with paired intervals excluding zero for all three seeds.
- The six-head audit separates radial law from graph structure for one seed:
  Indep-t versus Graph-t and Graph-G versus Graph-t are available under the same
  frozen representation contract, but their single-seed status remains explicit.
- The fixed-nu scan supports sensitivity of the radial-law result; it does not
  establish validation-selected nu or repair directional misspecification.
- Elasticity supports end-to-end trainability of the complete rank-4 path. Its
  three-seed averages are Full-t NLL `28.628 ± 0.490`, LR-t NLL `31.969 ±
  0.043`, Full-t Energy `7.187 ± 0.043`, and LR-t Energy `6.999 ± 0.011`.
  This is a trainability and family-trade-off diagnostic, not a claim that Full
  scatter dominates every proper metric.

## Rejected explanations / claims

- No six-head three-seed result is claimed; those cells were not jointly run.
- The temperature control is not reported as successful calibration: the
  validation-fitted global temperature changes test NLL from `-2.625` to
  `-2.426`, and block temperatures to `-2.385`.
- The fixed-nu scan is not a learned or validation-selected nu result.
- No external baseline, second group, five-seed ITOP factorial, E3 ensemble,
  mixture, or observation-aware path is added.

## Unresolved

- A true validation-selected global nu would require validation prediction
  artifacts or a new controlled training/evaluation protocol.
- Six-head multi-seed uncertainty-factorial variance remains unmeasured.
- All executable application evidence remains on the validated O(3) backend;
  no second-group empirical result is inferred.

## Elasticity representation-compatible follow-up

The formal elasticity evidence remains the legacy-Voigt deterministic/Full-t
stress test already recorded above.  A representation-compatible normalization
was checked separately as a feasibility diagnostic and was not promoted to a
headline result.  The seed-42 Full-t run remained finite for 12 epochs and
reached the complete 231-coordinate, ell=8 path.  The subsequent seed-43
stability audit is recorded in
`results/elasticity_stability_20260816`: shifted-log lowering with both
scalar and multiplicity normalization failed before the second epoch as the
unrestricted generator ran away.  A bounded centered spectral-window control
was finite and strict-SPD for 799 test structures, but changes the family
image and is therefore retained only as a control diagnostic.  Seed 44 was
not started.  No corrected-normalization or bounded-window metric is used in
the manuscript's formal comparison.

## Systems audit decision

The elasticity entry point already uses pinned transfer, non-blocking CUDA
copies, four workers, and persistent workers in the formal run.  Existing
batch-size timing records show a throughput improvement for the larger
measured batch, but changing batch size, precision, matrix-exponential map,
or compiler lowering would alter the numerical trajectory and was not
accepted as a semantics-preserving repair for the observed forward NaN.
The only retained code change is fail-fast rejection of non-finite training or
validation loss and non-finite gradient norms, with a regression test; finite
runs are otherwise unchanged.

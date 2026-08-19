# Documentation map

This directory is intentionally small at the current research boundary. The
root [`README.md`](../README.md) is the project overview; this page is the
working index for mathematical contracts, reproducibility rules, and
artifact-backed evidence.

## Start here

| Question | Current source |
| --- | --- |
| What does the compiler certificate mean? | [`compiler_certificate_scope.md`](compiler_certificate_scope.md) |
| What is the normative mathematical boundary? | [`../mathematical_contract.md`](../mathematical_contract.md) |
| What is the benchmark and selection contract? | [`../README.md#benchmark-contract--基准协议`](../README.md#benchmark-contract--基准协议) |
| What is the current experiment ledger? | [`journal_insight_experiment_ledger_20260814.md`](journal_insight_experiment_ledger_20260814.md) |
| What is the current dielectric family evidence? | [`dielectric_family_factorial_results_20260811.md`](dielectric_family_factorial_results_20260811.md) |
| What is the current elasticity evidence? | [`elasticity_end_to_end_results_20260811.md`](elasticity_end_to_end_results_20260811.md) |
| What is the current ITOP diagnosis? | [`itop_final_root_cause_diagnosis.md`](itop_final_root_cause_diagnosis.md) |
| What is the paired ITOP family/topology audit? | [`itop_paired_family_tradeoff_20260811.md`](itop_paired_family_tradeoff_20260811.md) and [`itop_reviewer_controls_evidence_20260817.md`](itop_reviewer_controls_evidence_20260817.md) |
| What is the statistical-misspecification conclusion? | [`statistical_misspecification_consolidation_20260812.md`](statistical_misspecification_consolidation_20260812.md) |

## Current research status

- The compiler, SPD, distribution, equivariance, and audit contracts are the
  maintained source of truth for the implementation.
- Existing formal dielectric, elasticity, and ITOP evidence remains separate
  by dataset, split, checkpoint, seed, and protocol. Numerical claims are not
  merged merely because they use the same metric.
- The Deep Ensemble external-control work has completed three independently
  initialized members from the compiled O(3) dielectric path (artifact source
  commit `5141d709903fe36c63b168f64f86a6981d9d0d60`). The exact density is an
  equally weighted Student-t member mixture. Its raw test mixture NLL is
  `-3.595` with Energy Score `0.405`; validation temperature calibration
  worsens these to `-3.190` and `0.419`. Coverage is reported only as the
  explicitly named `moment_gaussian_coverage`, so this remains an external
  diagnostic rather than a fixed-coordinate/non-equivariant baseline or a
  headline benchmark result. The metadata-only gate is
  `python scripts/audit_dielectric_ensemble_provenance.py`; it verifies clean
  source, shared dataset and inference-contract identity, staged training, and
  validation-to-test evaluation before the result is cited.
- The fixed-coordinate Cholesky smoke is retained as a numerical negative
  diagnostic. It is not a ranked baseline and is not being silently repaired
  or promoted.
- The exterior-square orientation refinement and shared-mean $K=2$ mixture are
  retained as diagnosis-driven negative pilots. They remain outside the public
  compiler-law surface because neither passed the predeclared adequacy gate.
- The registered `exp(asinh(lambda))` full-image chart has a complete audited
  three-seed representation-compatible elasticity result (mean NLL `18.809`,
  Energy `2.788`, MAE `13.333`); it is reported separately from the
  legacy-Voigt protocol and is not cross-protocol ranked.

## Reproducibility boundary

Training commands, environment identity, split rules, checkpoints, datasets,
and large prediction artifacts are kept with the experiment records and run
storage rather than copied into this source tree. A result is manuscript
eligible only when its artifact set and benchmark contract are complete. The
repository checks cover the CPU-testable compiler and metric paths; GPU and
dataset-dependent runs require their recorded environment and protocol.

## Archive

Completed pilots, superseded review ledgers, executed plans, and design notes
are preserved verbatim under [`archive/2026-08-18/`](archive/2026-08-18/). They
are historical provenance, not parallel current instructions. The archive
index records what was moved and why; use the current documents above when
making a new scientific or implementation decision.

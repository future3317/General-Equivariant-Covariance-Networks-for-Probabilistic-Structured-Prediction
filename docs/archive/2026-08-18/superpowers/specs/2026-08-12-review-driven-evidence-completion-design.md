# Review-Driven Evidence Completion Design

## Goal

Close only the reviewer-requested evidence gaps that can be answered from the
existing TPAMI artifacts or with a short, protocol-preserving post-processing
run. The manuscript will not promote unsupported training claims.

## Evidence boundaries

- Treat the ITOP six-head study as a single-seed factorial audit. It is not
  combined with the separate three-seed Full/LR/Graph robustness study.
- Treat the three-seed ITOP comparison as a paired Full-t/LR-t/Graph-t
  comparison using the existing prediction artifacts and exact frame IDs.
- Treat dielectric fixed-nu scanning as a post-hoc sensitivity diagnostic on
  saved predictions. It cannot be described as validation-selected training.
- Treat the existing validation-fitted global/block temperature result as a
  negative recalibration control because its held-out test NLL worsens.
- Use the existing Student-t likelihood, calibration, Energy Score, and
  prediction utilities. No second density implementation is allowed.
- Do not start new high-cost training, second-group experiments, external UQ
  baselines, E3b, mixture expansion, or observation-aware paths.

## Deliverables

1. A tested artifact-audit module that loads complete ITOP prediction cells,
   checks paired sample IDs and target equality, and emits six-head summaries
   plus bootstrap confidence intervals for predeclared family contrasts.
2. A tested dielectric post-hoc fixed-nu evaluator that reports NLL and
   existing diagnostics for the saved Full-t predictions at nu values
   `{3, 5, 10, 30}`. The output records that selection is not performed.
3. Compact JSON/Markdown evidence reports with provenance paths and explicit
   `existing evidence`, `new evidence`, `supported inference`, and
   `unresolved` sections.
4. A TPAMI manuscript update that only cites those artifacts, adds the
   complete single-seed ITOP factorial and paired three-seed intervals where
   space permits, reports the negative temperature control, and tightens
   claims about fixed-nu sensitivity and elasticity.

## Acceptance criteria

- All new code has unit tests written and observed failing before
  implementation.
- Bootstrap uses paired per-example differences, a fixed RNG seed, and an
  explicit percentile interval; no test labels are used for selection.
- Six-head rows are included only when the prediction artifact, metrics,
  provenance, and sample IDs are complete and finite.
- Fixed-nu output uses the existing production Student-t NLL semantics and
  records the source checkpoint/prediction hash.
- Existing dirty files in the code repository remain untouched.
- The manuscript compiles without undefined references/citations, passes
  `git diff --check`, and its rendered PDF has no clipped or unreadable tables.

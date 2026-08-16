# Journal Insight Closure Design

## Goal

Turn the current typed compiler study into a closed scientific loop: compile a
valid predictive law, diagnose law-level inadequacy with law-correct references,
replace only the failing law layer, and verify that algebraic guarantees remain
unchanged.

## Current context

The repository already contains partial implementations for conditional
Student-t, finite Student-t mixtures, representation-compatible elasticity
normalization, ITOP mixture diagnostics, spectral controls, and independent
compiler audits. The work therefore consolidates existing paths rather than
introducing parallel implementations.

## Design decisions

1. Dielectric conditional-nu will be evaluated on the three existing Full-t
   factorial checkpoints before any new law is promoted.
2. ITOP topology evidence will use a fixed split and paired true/shuffled
   topology runs. Random degree-matched trees will be generated before training
   and retained regardless of their results.
3. Elasticity will use a diagnostic-first protocol. Numerical stabilization is
   allowed only after the failing term and precision boundary are identified;
   learning-rate or seed-specific fixes are not accepted.
4. `PredictiveLaw`/`RadialLaw` will declare proper-score, sampling, moment,
   scatter/covariance, quantile, radial-reference, and diagnostic-reference
   semantics. Closed-form references are used when available; mixtures use
   simulation-based law-correct references.
5. The first non-elliptical prototype is K=2, shared mean, fixed nu=5, two
   equivariant Full scatters, and invariant mixture logits. Exact mixture NLL is
   logsumexp of component log densities; no moment matching or component-NLL
   averaging is permitted.
6. Mixture results enter the main paper only if proper score improves and the
   law-correct joint diagnostic explains the former single-ellipse failure.
7. The O(2)/SO(2) work is a feasibility audit and a minimal synthetic oracle
   case, not an application benchmark.

## Contracts and invariants

- Existing fixed-law configurations retain their current numerical path and
  results.
- New paths are explicit configuration choices and do not alter compiler
  equivariance, SPD construction, typed family reachability, or exact lowering.
- Validation controls checkpoint selection and hyperparameters; test is used
  only for final evaluation. ITOP Top is evaluation-only.
- All predictive artifacts record split, seed, config, compiler certificate,
  environment, objective semantics, and selection provenance.
- Every numerical repair is checked for finite loss, strict FP64 SPD,
  equivariance/basis consistency, and finite gradients before application runs.

## Decision gates

- Conditional-nu: retain if matched Full-t improves NLL and radial diagnostics
  without worsening the directional diagnostic; otherwise report checkpoint
  dependence.
- Topology: call the claim topology-specific only if true skeleton beats the
  pre-generated random-tree distribution under the fixed split; otherwise use
  the narrower graph-sparsity wording.
- Elasticity: promote only if all three predetermined seeds are finite under
  representation-compatible normalization and pass algebraic checks.
- Mixture: promote only with proper-score improvement plus a law-correct joint
  diagnostic improvement; NLL-only gains remain an appendix diagnostic.
- Second group: promote genericity only if the backend requires no compiler-core
  edits beyond representation/oracle/layout/lowering registration.

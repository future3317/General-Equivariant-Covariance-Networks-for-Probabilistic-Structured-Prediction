# Uncertainty-source semantics: minimal compiler design review

Date: 2026-08-09

Status: design only. No public schema or certificate behavior is implemented
by this document.

## Why this remains necessary

The compiler currently verifies representation reachability, operator typing,
SPD construction, family relations, and executor fidelity. Those guarantees do
not identify the stochastic source represented by a learned scatter. E0--E3
show that covariance geometry, radial law, latent modes, model diversity,
observation information, and training-objective coupling are separate failure
axes. A valid SPD matrix must not be silently described as physical aleatoric
covariance.

## Minimal source-level object

The smallest useful immutable record is a `StatisticalSemantics` value attached
to `DistributionSpec`, not a new task-specific compiler branch:

```text
StatisticalSemantics
  target: OutcomeConditional | ModelPredictive | ProtocolMixture |
          ObservationPosterior | SurrogatePredictiveGeometry
  observed_conditioning: tuple[str, ...]
  latent_variables: tuple[str, ...]
  identifiability_evidence: DeclaredEvidence
  calibration_scope: tuple[str, ...]
  excluded_sources: tuple[str, ...]
```

`DeclaredEvidence` should contain an evidence kind, protocol identifier, and
optional artifact hash. It records a user declaration; the compiler does not
infer scientific identifiability from a model or dataset name.

## Certificate semantics

Compilation reports should serialize the declaration and emit explicit
non-claims:

- algebraic/equivariant/SPD certificates do not prove calibration;
- they do not prove that the declared source is identifiable from the data;
- they do not convert Student-t scatter into physical covariance;
- missing repeated labels or latent-condition coverage remains an evidence
  limitation, not a lowering failure.

Only structural validation is in scope: known enum values, nonempty declared
conditioning, explicit calibration scope, and a declared evidence status.
Scientific truth remains an artifact-backed experimental claim.

## Migration and rejection policy

Existing public calls need an explicit legacy migration record such as
`SurrogatePredictiveGeometry` with `identifiability_evidence=Unspecified`; this
must be visible in the report and must not silently upgrade to
`OutcomeConditional`. New manuscript-facing experiments should be required to
provide an explicit record. An absent declaration may warn during a migration
window, but a contradictory declaration should fail planning with a typed
statistical-contract error rather than fall back to a different meaning.

## Reuse and non-goals

Reuse `DistributionSpec`, immutable planning records, compilation-report
serialization, and existing certificate/non-claim machinery. Do not add task
names, dataset names, calibration algorithms, mixture implementations, or
physical-domain heuristics. Implementation should wait for API review because
it changes the public source contract but requires no GPU experiment.

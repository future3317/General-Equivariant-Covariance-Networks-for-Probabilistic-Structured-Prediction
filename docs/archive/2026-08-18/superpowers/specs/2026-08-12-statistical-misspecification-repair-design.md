# Statistical Misspecification Repair Design

## Goal

Add opt-in predictive-law repairs for the existing frozen dielectric and ITOP
uncertainty-head protocols without changing compiler representation, SPD-map,
typed-family, or exact-lowering semantics.

## Design boundary

The compiler remains responsible for the existing equivariant feature,
operator-family, SPD, and lowering contracts.  New code is a distribution/head
composition above an already compiled `SPDMap`; each mixture component calls the
same certified map independently.  The legacy `fixed` path remains byte-for-
byte compatible at the model/config level and remains the default.

The first experimentable slice contains four frozen-head variants:

1. `fixed`: existing fixed `nu=5` baseline.
2. `global_nu`: one trainable invariant scalar with `nu > 2`.
3. `conditional_nu`: existing invariant feature readout with `nu > 2`.
4. `shared_mean_mixture`: K=2, shared frozen mean, component-specific typed
   scatter projections, invariant softmax weights, and component-specific
   bounded degrees of freedom.

`multimodal_mean_mixture` uses the same component machinery plus equivariant
offsets centered by the predicted mixture weights.  It is implemented and
tested but is not included in the first falsification command unless the
shared-mean mixture is non-degenerate.

## Probability semantics

`FiniteMixtureStudentTNLL` evaluates component log densities from each
component's SPD sufficient statistics and returns
`-torch.logsumexp(log_weight + component_log_prob, dim=0).mean()`.  It never
moment-matches components and never averages component NLLs.  Weights are
softmaxes of invariant scalar logits.  Component degrees of freedom use
`nu_min + softplus(raw)` with `nu_min > 2`; this preserves finite second-moment
diagnostics.  Sampling and calibration diagnostics accept scalar or
component/sample-valued degrees of freedom.

## Observation conditioning

Existing label-free point-cloud/depth descriptors are reused.  A new optional
descriptor payload is aligned by `sample_id` and concatenated only to the
invariant scalar input of the frozen uncertainty readout.  It cannot affect the
deterministic mean or compiler operator declaration.  Visibility and any
label-derived field are rejected from the conditioning path.

## Files

- `distributions/student_t.py`: preserve fixed-NLL behavior and expose shared
  normalized statistics helpers.
- `distributions/mixture.py`: exact K>=2 Student-t finite-mixture primitive.
- `distributions/__init__.py`: export the primitive.
- `models/frozen_distribution_readout.py`: add global-ν, invariant mixture
  logits, component scatter/mean heads, and optional invariant descriptors.
- `scripts/run_frozen_distribution_e1.py`: feature-gated variants, prediction
  serialization, exact mixture diagnostics, and descriptor alignment.
- `evaluation/ensemble.py`: backward-compatible component-valued sampling.
- `data/frozen_distribution_features.py`: optional descriptor field validation.
- `tests/test_statistical_misspecification_repair.py`: law, invariance, SPD,
  exact-mixture, centering, and descriptor-gating tests.
- `scripts/prepare_itop_observation_descriptors.py`: build aligned optional
  descriptor payloads from existing geometry/depth caches.

## Acceptance gates

All existing tests must pass.  New tests must prove fixed-NLL numerical
compatibility, invariant ν/logit/offset behavior under O(3), finite ν>2,
finite SPD components, exact logsumexp semantics, weighted mean preservation,
descriptor sample-id alignment, and rejection of visibility/label fields.

# E1 frozen-distribution protocol

Date: 2026-08-08

Phase gate: E0 in `uncertainty_root_cause_audit_20260808.md` rejects the fixed-nu
single ellipse for dielectric Full-t and ITOP Graph/Full-t. E1 changes only the
conditional distribution readout. E2, E3, temperature calibration, conformal
calibration, and manuscript-result updates are out of scope.

## Question and controlled evidence

### Distribution-law factorial

- **Hypothesis:** fixed radial-tail behavior or single-component topology, rather
  than missing SPD coordinates, causes a material part of held-out failure.
- **Existing evidence:** Full-t still fails E0; matching-family synthetic recovery
  succeeds; no conditional-nu or trained K=2 artifact exists.
- **Frozen evidence:** one released Full-t checkpoint supplies immutable typed
  features H, mean mu, shared scatter parameters, split IDs, and hashes.
- **Baseline:** exact fixed-nu=5 Student-t evaluation from the frozen artifact.
- **Conditional-nu intervention:** train only one `0e` readout with
  `nu(x)=2.05+softplus(raw(x))`; mean and scatter remain byte-identical inputs.
- **K=2 intervention:** train only one equivariant offset readout and evaluate
  component means `mu +/- delta`, fixed weights `(1/2,1/2)`, fixed nu=5, and the
  same frozen shared compiled scatter.
- **Controlled variables:** split IDs, seed policy, H, mu, scatter, optimizer,
  FP32 operator/NLL algebra, early stopping, and validation-only epoch selection.
- **Decisive outcome:** conditional nu improves radial fit but is not expected to
  repair directional structure; K=2 must improve held-out exact mixture NLL and
  mixture-aware projection PIT/score, not merely one coverage number.
- **Interpretation boundary:** a K=2 gain supports a topology intervention but
  does not identify physical aleatoric uncertainty or epistemic uncertainty.

The first K=2 stage deliberately does not learn gates, conditional nu, or
component-specific scatters. Those freedoms are forbidden until the shared-
scatter symmetric intervention shows stable proper-score benefit.

### Dielectric matched spectral control

- **Hypothesis:** the centered shape window, rather than density topology, is the
  principal cause of dielectric failure.
- **Existing evidence:** old spectral checkpoints are unmatched and the released
  centered run reaches its theoretical condition-number boundary.
- **Intervention:** from one frozen H and mu, reset the same typed Full operator
  projection identically and train it under only one of: centered shape window
  `[-2,2]` (kappa <= exp(4)), a preregistered wider centered window `[-4,4]`
  (kappa <= exp(8)), or unbounded matrix exponential.
- **Controlled variables:** train/validation/test IDs, seed, projection
  architecture and initialization, optimizer, FP32 algebra, patience, and
  validation-NLL selection.
- **Decisive outcome:** the restriction is causal only if wider/unbounded maps
  reproducibly improve held-out NLL and shape diagnostics without changing H,
  mu, or distribution law.

## Evaluation semantics

- Single-component models: exact Student-t NLL plus E0 radial/projection/
  direction/radius-direction diagnostics.
- K=2: exact weighted component `logsumexp` NLL, Student-t mixture Energy Score,
  and random-direction PIT using the weighted component projection CDF. A
  moment-matched Student-t is prohibited. Single-ellipse whitening tests are not
  success criteria for the mixture.
- ITOP selection uses only the declared Side validation split. Side test is IID
  evaluation and Top is a preregistered cross-view OOD evaluation; Top never
  changes epoch, hyperparameters, or model choice.

## Required artifacts

Every run must atomically publish:

- `protocol.json`: hypothesis, intervention, frozen hashes, exact split/ID
  hashes, distribution/operator schemas, seed, optimizer, precision, and NLL
  semantics;
- `history.json`: train/validation losses, gradient norms, and selected epoch;
- `best_model.pt` and its SHA-256;
- split prediction tensors and their SHA-256 records;
- `diagnostics.json`: proper scores and law-appropriate calibration diagnostics;
- `environment.json`: clean source commit, device, dtype, dependency versions,
  and cache/checkpoint chain.

## Stop rule

If K=2 lacks stable held-out proper-score and mixture-projection-calibration
improvement under this control, stop mixture expansion and move to E2. A
positive K=2 result permits design of a public compiler-level finite-mixture
primitive; this E1 composition itself is not presented as that primitive.

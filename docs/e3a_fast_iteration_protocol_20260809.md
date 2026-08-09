# E3a fast-iteration protocol (2026-08-09)

## Status and purpose

This is a development-only falsification panel, not a paper result.  The first
full-data, 512-point, single-stage end-to-end member improved validation NLL to
about -97 while validation MPJPE remained about 67.6 cm.  That run is retained
as negative optimization evidence and is not an E3 ensemble member.

## Hypothesis

Independent models can only test model/function uncertainty after each member
has learned a useful mean function.  Training the mean first within each seed,
then fitting and fine-tuning that seed's own uncertainty head, should prevent
the scatter head from absorbing the objective before the mean is established.

## Existing evidence

- E0--E2 leave model/function uncertainty as the remaining live explanation.
- The single-stage E3a member failed the mean-sufficiency prerequisite.
- The repository already implements deterministic, frozen-head, feature-cache,
  and joint-fine-tune stages with checkpoint, RNG, split, and cache provenance.

## Exact intervention

Use the existing `scripts.run_itop_study` stages with only `full_student_t`:

- Side-train exposure: 2,487 samples (explicit development cache).
- Side/Top test: all 4,863 valid samples; Top never selects a checkpoint.
- Points: 256; neighbors: 16.
- Seeds: 42 and 43.
- Shared train/validation split seed: 42.  Model seeds affect initialization,
  shuffle, and worker RNGs, but not sample membership.
- Stage 1: independent deterministic training, at most 8 epochs, patience 2.
- Stage 2: each seed's own frozen Full Student-t head, fixed nu=5, at most
  5 epochs, patience 2.
- Stage 3: only after the Stage-2 gate, joint-fine-tune each same-seed member
  for at most 3 epochs, patience 2.

No backbone, mean checkpoint, feature cache, or uncertainty head is shared
between seeds.  The reduced data and point count are an algorithmic pilot
configuration and must not be presented as the final E3 result.

## Controlled variables

Architecture, optimizer, likelihood, fixed nu, FP32 operator/NLL algebra,
BF16 backbone autocast, split policy, full Side/Top evaluation sets, and exact
checkpoint selection semantics remain fixed across the two pilot seeds.

## Phase gates

1. **Mean prerequisite:** both deterministic members must have finite artifacts
   and Side MPJPE no worse than 35 cm.  Failure stops E3; do not train more
   uncertainty heads.
2. **Frozen distribution pilot:** both Full-t heads must have finite FP64 scale
   certificates.  Evaluate the two-member equal-weight density with the
   existing exact finite-mixture `logsumexp`, Energy Score, projection PIT, and
   between-member spread code.  No moment-matched pseudo-NLL is allowed.
3. **Joint pilot:** run only if Gate 2 shows a proper-score or projection benefit
   and nontrivial mean diversity.  Reject joint fine-tuning if it materially
   degrades the deterministic mean or cross-view behavior.
4. **Full confirmation:** only a positive two-seed pilot permits full Side-train,
   512-point, three-seed confirmation.  Otherwise stop E3 and record the
   negative result.

## Interpretation boundaries

The pilot can reject an unpromising E3 protocol quickly.  It cannot establish
final robustness, calibration, or physical aleatoric covariance.  Between-seed
spread remains a model/function diagnostic.

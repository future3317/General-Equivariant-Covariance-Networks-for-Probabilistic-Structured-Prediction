# Finite-mixture compiler design review (implementation deferred)

This is an interface review only. E2 did not pass the distribution-topology
gate, so no public compiler primitive is implemented or claimed.

## Current reusable pieces

- `equivcompiler.distributions.DistributionSpec` and `EllipticalDistribution`
  already separate representation/operator semantics from the radial law.
- `representations.operator_ir.OperatorFamilyPlan` describes each component's
  typed operator program, SPD/equivariance certificate, and computational
  oracles.
- `CompiledProbabilisticReadout` and the existing projection/lowering path can
  produce equivariant locations and certified SPD scatter parameters.
- `evaluation.ensemble.finite_mixture_log_prob` already implements exact
  weighted `logsumexp`, component log probabilities, and posterior
  responsibilities without moment matching.
- `evaluation.ensemble.sample_ensemble` and
  `energy_score_from_samples` already provide sampling and proper-score
  evaluation for finite mixtures.
- Conditional Student-t broadcasting and mixture-aware projection PIT already
  exist in the E1 evaluation path.

## Minimal future schema

If a later phase supports mixture topology, the smallest coherent extension is
a distribution composition rather than a task/model-name branch:

1. **Distribution schema**
   - `FiniteMixtureDistribution(component: DistributionSpec, K: int)`;
   - explicit weight semantics (`fixed` or invariant learned logits);
   - exact-density and sampling oracle declarations;
   - no moment-matched fallback.
2. **Typed readout schema**
   - component locations with shape `(K, ..., dim(V))`, each typed as `V`;
   - mixture logits typed only as copies of `0e` and normalized by softmax;
   - one shared or K component operator parameter programs, declared explicitly;
   - component radial parameters, when conditional, restricted to legal
     invariants.
3. **Execution contract**
   - lower every component location through the existing equivariant readout;
   - lower every component scatter through an existing certified operator/SPD
     program;
   - evaluate exact `logsumexp(log_weight + component_log_prob)`;
   - sample a categorical component followed by that component's existing
     sampling oracle.
4. **Checkpoint/provenance contract**
   - serialize K, weight policy, component-sharing policy, radial semantics,
     operator-program hashes, executor certificates, and exact NLL semantics;
   - reject checkpoint restoration when any component layout or sharing policy
     differs.

## Certificate semantics

A valid mixture certificate would need to establish only the following:

- every component location transforms in `V`;
- learned weights are invariant, positive, and sum to one;
- every component scatter is SPD and equivariant under its operator program;
- the resulting probability measure has the required pushforward equivariance;
- density evaluation uses exact finite-mixture `logsumexp` and sampling uses the
  declared categorical/component oracle.

It must explicitly **not** certify calibration, physical multimodality,
identifiability, component semantics, or superiority over a single component.
Component label permutations are equivalent and should be treated as such in
checkpoint comparisons and diagnostics.

## Deferred choices

- fixed versus learned invariant weights;
- shared versus component-specific scatter programs;
- symmetric offsets versus unrestricted equivariant component locations;
- shared versus component-specific radial parameters;
- whether compiler reachability is checked per unique shared program or per
  component instance.

These choices should not enter the public API until a controlled experiment
shows that exact mixture topology improves held-out proper scores beyond a
validation-selected single-location correction. E2 showed the opposite on
ITOP Side, so implementation is deferred.


# Dielectric frozen-family factorial design

## Status and scientific question

The independent NumPy/SciPy recovery gate passed for all four families and
three seeds. This specification therefore defines the next P0 experiment:

> With exactly the same frozen dielectric representation `H`, mean `mu`, data
> splits, optimizer, and selection protocol, how do covariance family and
> radial law independently affect held-out proper predictive performance?

This is a frozen-head family/law test, not a new backbone experiment and not an
attempt to identify physical or DFPT aleatoric covariance from single labels.
No manuscript result changes are authorized until the formal factorial is
complete and audited.

## Reused evidence and components

The immutable E1 dielectric cache at
`RESULTS/Tpami/E1/frozen_distribution_96a4975/dielectric_unified_full_cache`
contains one frozen feature representation and mean for 4,236 train, 485
validation, and 281 test samples. The exact split tensors, sample IDs, source
checkpoint, dataset metadata, and source hashes are already recorded.

The implementation must reuse:

- `frozen_distribution_loaders` for cached `H`, `mu`, targets, and IDs;
- `plan_readout` and the existing typed operator-family policies;
- `GaussianNLL` and `StudentTNLL` for production likelihoods;
- existing Energy Score, projection-PIT/elliptical, coverage, and spectrum
  diagnostics;
- atomic JSON, checkpoint, prediction, and SHA-256 provenance utilities.

No second Gaussian/Student-t density, SPD map, calibration routine, data split,
or dielectric backbone trainer may be introduced.

## Factorial arms

The eight arms are the Cartesian product of four operator families and two
radial laws:

| Label | Typed compiler policy | Active scatter coordinates |
|---|---|---:|
| Isotropic | `LowRankCovariance(rank=0)` | 1 |
| Block | `IsotypicBlockCovariance()` | 2 |
| Low-rank | `LowRankCovariance(rank=2)` | 13 |
| Full | `CenteredSpectralWindowCovariance(-2,2,-8,8)` | 21 |

The laws are Gaussian and Student-t with fixed `nu=5`. The centered Full map is
the current released Full family. E1's matched spectral control already found
that widening or removing its spectral window was not the principal
dielectric failure, so map variants are not repeated here.

All arms use the same cached feature irreps and frozen mean. Each arm trains
only the compiler-declared operator projections. A generic frozen-mean
elliptical readout may compose an existing operator program with the existing
Gaussian or Student-t objective; it must not contain family-name conditionals
or duplicate probability algebra.

## Controls and selection

Formal training seeds are `42,43,44`. The cached dataset and split hashes are
identical across every arm and seed. Seeds change only projection
initialization and train-loader order.

Common controls are:

- batch size 128;
- Adam, learning rate `5e-4`, weight decay `1e-5`;
- at most 60 epochs;
- `ReduceLROnPlateau(factor=0.5, patience=2)`;
- early-stopping patience 5;
- FP32 cached features, operator assembly, and NLL algebra;
- validation NLL as the only checkpoint-selection quantity.

Test metrics may never change an epoch, hyperparameter, seed, family, or law.
The Full/Student-t arm is not initialized from the old Full projection: every
factorial arm receives a fresh operator projection under its own seed. Frozen
means remain byte-identical across all arms.

## Fast staged execution

### Stage 0: compile and unit smoke

On CPU, compile all eight schemas and run forward/backward, finite, SPD,
equivariance, and exact-objective reuse tests on small synthetic batches.

### Stage 1: one-seed operational pilot

Run all eight arms with seed 42, the complete cached splits, at most 20 epochs,
and patience 5. This pilot is development-only and answers whether the matched
protocol is operational. It passes only if:

- all arms compile with an exact eligible executor and the requested family;
- all losses, predictions, and scatters are finite and strictly SPD in the
  declared evaluation dtype;
- all arms use identical cache/split/mean hashes and validation-only selection;
- every required artifact and recorded hash verifies;
- the freshly trained Full/Student-t test NLL is within 0.20 nat/sample of the
  existing frozen E1 reference (`-2.6247`), a coarse harness-regression check.

Failure stops the factorial for harness diagnosis. Learner results may not be
used to relax this operational gate.

### Stage 2: formal three-seed factorial

Only after Stage 1 passes, run seeds 42, 43, and 44 with the formal 60-epoch
budget. The formal run includes a fresh seed-42 arm under the formal output
contract; the development pilot is not silently reused as formal evidence.

## Evaluation and decisive outcomes

Every arm reports held-out exact NLL including normalization constants, Energy
Score, Coverage90/95, random-projection PIT/calibration, radial and directional
single-ellipse diagnostics, spectrum/condition diagnostics, and selected
epoch. Gaussian diagnostics use Gaussian references; Student-t diagnostics use
`q/d ~ F(d,nu)` and Student-t marginal quantiles.

Primary comparisons are predeclared:

1. within each operator family, Student-t minus Gaussian paired test NLL;
2. within each radial law, each restricted family minus Full paired test NLL;
3. the best validation-selected family/law versus Full Student-t on test NLL,
   with Energy Score and projection calibration as required corroborating
   metrics.

Report every seed and mean plus standard deviation. Bootstrap over held-out
samples may quantify paired metric uncertainty, but test bootstrap results may
not select the arm. A prettier Coverage90/95 value alone is not a success.

Interpretation is limited to:

- a stable law effect if Student-t improves proper NLL and at least one
  corroborating score/calibration metric across families and seeds;
- a stable family effect if a restricted family improves proper NLL and at
  least one corroborating metric without changing `H` or `mu`;
- no family-resolution evidence if differences are seed-sensitive or appear
  only in marginal coverage;
- persistent directional rejection as evidence that changing operator
  coordinates or radial law does not repair the full predictive law.

## Artifact contract

Each arm saves `args.json`, `schema.json`, `environment.json`, atomic
`history.json`, best/last checkpoints, test predictions, `metrics.json`, and a
diagnostic JSON. Provenance records source commit/dirty state, cache metadata
and split hashes, source checkpoint hash, seed, optimizer/selection protocol,
compiler/operator/distribution schema, selected epoch, finite/SPD checks, and
all artifact hashes.

The aggregate result verifies that frozen means and sample IDs are identical
across arms before computing comparisons. Internal hashes remain artifact
provenance rather than manuscript prose.

## Scope boundary

This protocol does not train a new backbone, conditional `nu`, mixtures,
ensembles, conformal/temperature calibration, observation-aware paths,
elasticity models, or graph kernels. It does not alter the public compiler
schema except for any genuinely missing task-neutral composition needed to
bind an existing operator program to an existing elliptical objective.

# Independent NumPy/SciPy Teacher Recovery Design

## Status and decision

This specification defines the P0 controlled scatter-recovery experiment that
must precede the dielectric family factorial. The chosen design is a minimal
independent NumPy/SciPy oracle integrated with the existing learner-side
compiler benchmark, with immutable dataset and manifest artifacts.

The isolation principle is:

> **Teacher defines the data-generating distribution independently; the
> compiler is used only on the learner side.**

The experiment answers one question: can a compiler-generated learner recover
known Full, low-rank-plus-isotropic, isotypic-block, and graph-precision
Student-t scatter families when the data-generating distribution was not
constructed by the compiler or its runtime maps?

This experiment does not test real-data calibration, identify physical
aleatoric covariance, or compare neural backbones.

## Scope

### Included

- An oracle-only NumPy/SciPy implementation of the complete teacher path:
  input generation, zero-mean contract, scatter or precision construction,
  Student-t sampling, rotations, and oracle self-checks.
- Full, rank-2-plus-isotropic, isotypic-block, and graph-precision teachers.
- Compiler-generated learners using the existing public planning API.
- Matched recovery for all four families.
- A shared-dataset Full/LR/Block cross-family matrix.
- Three independent dataset/training seeds.
- Validation-only learner selection and immutable provenance artifacts.
- Pre-learner teacher-only Monte Carlo calibration of statistical tolerances.
- An optional test-only SciPy Student-t NLL reference used solely to validate
  the existing production `StudentTNLL` implementation on fixed arrays.

### Excluded

- A second production NLL, SPD map, distribution class, or compiler path.
- Graph cross-family comparisons, because its output representation and
  structural domain differ from the rank-2 Full/LR/Block group.
- Dielectric, ITOP, elasticity training, certificate-checker work, and graph
  performance optimization.
- Any manuscript update before the independent recovery gate passes.

## Independence boundary

The oracle module is import-isolated. It may import only Python standard-library
modules, NumPy, and SciPy. It must not import PyTorch, e3nn, `equivcompiler`,
`plan_readout`, model classes, `spd_maps`, `distributions`, or evaluation
utilities from this repository.

The oracle computes every data-generating quantity in `float64`. Array
conversion to PyTorch occurs only after the `.npz` dataset and JSON manifest
have been finalized and hashed. The learner runner is allowed to import the
oracle's serialized arrays, but the oracle may not call into the learner.

Student-t observations are sampled independently as

\[
y=\mu+Lz\sqrt{\nu/u},\qquad
z\sim\mathcal N(0,I),\quad u\sim\chi^2_\nu,
\]

using NumPy random generators and SciPy linear algebra. Here `L` is the
Cholesky factor of scatter `S`; `S` is not mislabeled as covariance. The
experiment uses fixed `nu=5`, so `Cov(Y|x)=nu/(nu-2) S` exists.

The only shared object between oracle and learner is the declared coordinate
contract. For `0e+2e`, the oracle contains an explicit, documented orthonormal
scalar-plus-STF basis expressed with square-root constants. It does not obtain
that basis from e3nn at runtime. A unit test checks that this public coordinate
contract agrees with the repository's `e3nn_real_v1` convention; this check is
not part of data generation.

## Independent family constructions

All teacher coefficients are fixed by an oracle version and seed before any
learner is trained. Coefficient magnitudes keep scatter condition numbers in a
declared numerically comfortable range; no runtime clipping or fallback is
permitted.

### Full scatter on `V = 0e + 2e`

An input context is `(s,T)`, where `s` is invariant and `T` is a Cartesian
symmetric trace-free tensor. The oracle builds a dense symmetric log-scatter
operator on `(a,B) in 0e+2e` from direct Cartesian contractions:

- invariant scalar-to-scalar and STF-to-STF identity terms;
- scalar/STF cross terms proportional to `T`;
- the equivariant STF map
  `B -> STF(TB + BT)`;
- a quadratic `t t^T` term in orthonormal STF coordinates.

The matrix exponential is evaluated with `scipy.linalg.expm`. The construction
generically contains scalar/STF coupling and non-isotypic STF shape, so it is
not an isotypic-block teacher and is not constrained to rank two plus an
isotropic residual.

### Rank-2 plus isotropic scatter on `V = 0e + 2e`

The oracle constructs two equivariant columns from `(s,T)`: one from `(s,T)`
and one from invariant scalars plus `STF(T^2)`. It then evaluates

\[
S(x)=\sigma^2(x)I_6+L(x)L(x)^\top,
\]

with a strictly positive invariant `sigma^2(x)`. This is exactly the declared
rank-2-plus-isotropic statistical family; no compiler low-rank map is called.

### Isotypic-block scatter on `V = 0e + 2e`

Because both irreps have multiplicity one, the oracle uses

\[
S(x)=\operatorname{diag}(k_0(x),k_2(x)I_5),
\]

where `k_0` and `k_2` are positive invariant functions of `s` and
`tr(T^2)`. This is exactly the complete input-dependent operator family that
commutes with the `0e+2e` action for multiplicity-one blocks.

### Graph precision on three `1o` nodes

The graph is the chain `(0,1),(1,2)`. Inputs are three Cartesian vectors.
Unary and edge log-precision blocks are direct equivariant functions of node
vectors and edge differences. SciPy matrix exponentials produce local SPD
blocks `U_j` and `W_e`, and NumPy assembles

\[
Q=\operatorname{BlockDiag}(U)
 +(B\otimes I_3)^\top\operatorname{BlockDiag}(W)(B\otimes I_3),
\qquad S=Q^{-1}.
\]

The oracle saves both `Q` and `S`. The primary Graph recovery metric is relative
precision error, with scatter error reported as a secondary check.

## Dataset protocol

For each teacher family and seed, the oracle writes one immutable `.npz`
dataset before learner construction. The file contains train, validation,
test, and teacher-only calibration context IDs; inputs; observations; means;
true scatter; and, for Graph, true precision.

The default formal protocol is:

- seeds: `0,1,2`;
- train contexts: `128`, with `32` repeated observations per context;
- validation contexts: `64`, with `64` repeated observations per context;
- test contexts: `128`, with `128` repeated observations per context;
- teacher-only calibration draws: `65,536` per family and seed;
- Student-t degrees of freedom: `5`;
- all oracle arrays: `float64`;
- learner operator/NLL algebra: FP32 for training and FP64 materialization for
  formal scatter/precision evaluation.

The same `.npz` file and hash must be consumed by every Full/LR/Block learner
for a given teacher family and seed. Therefore Full-teacher comparisons among
Full, LR, and Block learners have identical inputs, observations, split IDs,
and sampling noise; the same rule applies to LR and Block teachers. Graph uses
its own representation-specific dataset and matched learner only.

Learner mean is fixed to the oracle's explicit zero mean. Only scatter-head
parameters are trained. Existing `StudentTNLL`, compiler operator assembly,
optimizer primitives, and evaluation metrics are reused. Checkpoint selection
uses validation NLL only; test metrics cannot affect step count, learning rate,
or selection.

## Predeclared gates

No threshold may be changed after reading learner results.

### Numeric gates

Oracle self-checks use FP64:

- minimum scatter and precision eigenvalue: greater than `1e-10`;
- maximum relative equivariance error for mean/scatter/precision: `5e-10`;
- maximum orthogonal-coordinate relative scatter error: `5e-10`;
- maximum orthogonal-coordinate absolute NLL discrepancy: `5e-10`.

Learner checks account for FP32 training and FP64 evaluation materialization:

- finite parameters, predictions, scatter/precision, and NLL: required;
- maximum relative equivariance error: `5e-5`;
- maximum orthogonal-coordinate relative scatter error: `5e-5`;
- maximum orthogonal-coordinate absolute NLL discrepancy: `5e-5`.

### Recovery gates

Every matched family and every seed must satisfy:

- mean relative Frobenius error of the primary operator no greater than `0.05`;
- 90th-percentile per-context relative Frobenius error no greater than `0.10`;
- for Graph, the primary operator is precision `Q`; scatter `S` must also have
  mean relative error no greater than `0.075`;
- validation-selected checkpoint exists and all artifact hashes verify.

The Full/LR/Block cross-family rows are diagnostic and receive no pass/fail
recovery threshold. They must preserve the common dataset hash and report the
same metrics as matched rows. A cross-family learner outperforming a matched
learner is reported as evidence against the intended family construction or
training harness, not silently discarded.

### Coverage90/95 statistical gate

Before learner construction, the oracle calibration stream computes exact-law
Mahalanobis coverage at nominal `0.90` and `0.95` using
`q/d ~ F(d,nu)`. For each nominal level and evaluation sample count, the oracle
then simulates paired exact-model coverage estimates and stores the 99% upper
quantile of their absolute difference as `sampling_tolerance`. This tolerance
is frozen in the manifest.

A matched learner passes coverage only when its held-out Coverage90 and
Coverage95 differ from the corresponding oracle held-out estimates by no more
than the frozen paired-Monte-Carlo tolerance. This gate measures agreement with
the independently generated predictive law; it is not tuned from learner
coverage and does not rely on an arbitrary post-hoc percentage allowance.

All four matched families and all three seeds must pass numeric, recovery, and
coverage gates before the dielectric family factorial can start.

## Artifact and provenance contract

Each oracle dataset produces:

- `<family>_seed_<seed>.npz`;
- `<family>_seed_<seed>.manifest.json`;
- SHA-256 for both files;
- oracle schema/version and source commit;
- clean/dirty source state recorded by the runner;
- family equations and fixed coefficient record;
- seed and NumPy bit-generator state identifier;
- split sizes and exact context-ID hashes;
- dtype, `nu`, graph schema where applicable;
- teacher-only gate calibration and frozen tolerances;
- finite/SPD/equivariance/invariance self-check results.

Each learner row additionally records learner family, public compiler report,
dataset and manifest hashes, optimizer/selection protocol, selected epoch,
checkpoint hash, prediction hash, and diagnostic metrics. Result JSON is
written atomically.

These details support artifact audit only. If the gate passes, the manuscript
protocol will state that the teacher was an independently implemented
NumPy/SciPy oracle, name the families, sample sizes, seeds, fixed `nu`, learner
compiler path, and validation-only selection. Internal hashes and debug checks
will remain in the artifact supplement rather than the main paper.

## Code organization

- `experiments/independent_teacher_oracle.py`: NumPy/SciPy-only teacher,
  family equations, sampling, oracle checks, `.npz` and manifest writing.
- `experiments/synthetic_covariance_benchmark.py`: retain learner-side compiler,
  NLL, optimization, metrics, and CLI; add an independent-dataset mode and
  remove the compiler teacher from the formal evidence path.
- `tests/test_independent_teacher_oracle.py`: import-isolation, family math,
  equivariance, SPD, reproducibility/hash, shared-dataset, and SciPy reference
  checks.
- `tests/test_synthetic_covariance_benchmark.py`: learner consumption,
  validation-only selection, provenance, and gate regression tests.
- `docs/independent_teacher_recovery_results_20260811.md`: generated only after
  the formal three-seed run, with existing evidence, new evidence, supported
  inference, rejected explanation, and unresolved items separated.

The existing plotting scripts are outside the implementation scope until the
formal gate passes. Existing uncommitted user changes in those files must not
be staged or modified.

## Execution and decision sequence

1. Add failing oracle and runner contract tests.
2. Implement the minimum NumPy/SciPy oracle and learner adapter.
3. Run local unit and smoke tests.
4. Generate teacher-only calibration manifests and freeze all tolerances.
5. Commit and push the implementation.
6. Create a clean server worktree at the exact commit and activate
   `equivcompiler` with the declared data environment.
7. Run the three-seed independent recovery benchmark.
8. Download datasets, manifests, logs, checkpoints, result JSON, and hashes.
9. Verify all gates locally and write the evidence document.
10. Only if every matched row passes, design and launch the dielectric family
    factorial. If matched recovery is unstable, stop and repair the recovery
    harness without using dielectric results to tune it.

## Success interpretation

A passing result supports the narrow claim that the learner-side compiler can
recover these controlled scatter/precision families when the data-generating
distribution is independently implemented. It removes the current circular
teacher/planner validation concern.

It does not prove calibration on real scientific data, completeness for all
equivariant SPD fields, or correctness of any physical uncertainty
interpretation.

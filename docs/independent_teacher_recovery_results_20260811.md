# Independent NumPy/SciPy teacher recovery results

Date: 2026-08-11

Status: **formal gate passed (12/12 matched family-seed rows).** The dielectric
family factorial is permitted by the predeclared protocol, but has not been
started as part of this experiment.

## Existing evidence

The previous controlled scatter-recovery benchmark used a teacher and learner
whose operator programs were both produced by the public compiler planner.
That experiment tested optimization and recovery behavior, but it could not
exclude a shared implementation error in family planning or SPD assembly.

The predeclared replacement protocol therefore imposed the isolation rule:

> **Teacher defines the data-generating distribution independently; the
> compiler is used only on the learner side.**

The complete teacher path--input generation, zero mean, scatter or precision
construction, and Student-t sampling--was implemented using only NumPy and
SciPy in FP64. It did not import the compiler, planner, PyTorch, e3nn, the
production SPD maps, or the production likelihood. Learners continued to use
the public compiler and the existing `StudentTNLL`; no second production
likelihood was introduced.

## New evidence

The formal run used fixed `nu=5`, three seeds, 128/64/128 train/validation/test
contexts, 32/64/128 observations per context, validation-NLL-only checkpoint
selection, and FP64 operator materialization for evaluation. Full, low-rank,
and isotypic-block learners consumed the identical serialized teacher dataset
within each teacher-family/seed comparison. Graph precision used its matched
three-node chain dataset.

All matched rows passed the predeclared numerical, recovery, coverage, and
artifact-provenance gates:

| Teacher / learner | Seed | Selected epoch | Primary mean rel. error | Primary p90 rel. error | Scatter mean rel. error | Cov90 | Cov95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 0 | 30 | 0.02952 | 0.05690 | 0.02952 | 0.89819 | 0.94910 |
| Full | 1 | 98 | 0.04707 | 0.07923 | 0.04707 | 0.90009 | 0.95117 |
| Full | 2 | 17 | 0.04182 | 0.07478 | 0.04182 | 0.89673 | 0.94861 |
| Low-rank | 0 | 61 | 0.01796 | 0.03474 | 0.01796 | 0.90204 | 0.95117 |
| Low-rank | 1 | 78 | 0.01744 | 0.03037 | 0.01744 | 0.89838 | 0.95026 |
| Low-rank | 2 | 82 | 0.02071 | 0.04211 | 0.02071 | 0.90338 | 0.95111 |
| Isotypic block | 0 | 71 | 0.00590 | 0.01155 | 0.00590 | 0.90216 | 0.95142 |
| Isotypic block | 1 | 256 | 0.02360 | 0.04887 | 0.02360 | 0.90106 | 0.95148 |
| Isotypic block | 2 | 194 | 0.00688 | 0.01224 | 0.00688 | 0.90265 | 0.95068 |
| Graph precision | 0 | 36 | 0.04870 | 0.08981 | 0.03493 | 0.90674 | 0.95282 |
| Graph precision | 1 | 51 | 0.04337 | 0.07149 | 0.04398 | 0.90424 | 0.95099 |
| Graph precision | 2 | 62 | 0.04850 | 0.07709 | 0.04462 | 0.90363 | 0.95081 |

For Graph, the primary operator is precision; the reported scatter error is a
secondary inversion check. The largest primary mean error was 0.04870, close
to but below the frozen 0.05 threshold. The largest primary p90 error was
0.08981, below 0.10. All learner equivariance errors were at most
`2.38e-7`, versus the FP32-aware limit `5e-5`.

The 18 shared-dataset cross-family diagnostic rows behaved as intended. For
every Full, low-rank, and isotypic-block teacher seed, the matched learner had
both the lowest held-out NLL and the lowest primary operator error among the
three learners. No cross-family learner unexpectedly outperformed the matched
family.

## Artifact audit

- clean source commit:
  `bc0d297c120d0fd954f116fb93c84fc0c5344223`;
- formal output: 12 matched rows and 18 cross-family diagnostic rows;
- all predictions, NLLs, scatters, and precisions finite;
- all 60 checkpoint/prediction references and all 24 oracle NPZ/manifest
  references re-hashed successfully after transfer;
- every row records validation-only selection;
- every oracle manifest records the same clean source commit, FP64 teacher
  self-checks, split hashes, teacher coverage, and frozen Monte Carlo coverage
  tolerances;
- Full/low-rank/isotypic-block comparisons share exactly one NPZ and manifest
  hash for each teacher-family/seed pair;
- all oracle FP64 equivariance errors are below `5e-10`, and all minimum
  scatter eigenvalues exceed `1e-10`;
- learner orthogonal-coordinate scatter and NLL invariance checks pass their
  predeclared `5e-5` limits.

Evidence paths:

- server:
  `/home/workspace/lrh/RESULTS/Tpami/Synthetic/independent_teacher_bc0d297`;
- local:
  `results/independent_teacher_recovery_bc0d297`;
- design:
  `docs/archive/2026-08-18/superpowers/specs/2026-08-11-independent-teacher-recovery-design.md`.

## Supported inference

The public compiler learner recovers independently generated Full,
low-rank-plus-isotropic, isotypic-block, and graph-precision Student-t
operators under the controlled protocol. Because the teacher does not use the
planner or production operator maps, the result removes the principal
circularity in the former synthetic recovery evidence.

This supports a manuscript statement that controlled family recovery was
validated against an independently implemented NumPy/SciPy oracle. It does
not establish real-data calibration, physical aleatoric covariance, or that a
particular family is appropriate for dielectric data.

## Rejected explanation

The matched recovery result is not explained by teacher and learner sharing
the same planner, SPD map, or runtime likelihood implementation. The
cross-family results also provide no indication that the independent teacher
families collapsed to an unintended common family.

## Unresolved and gate decision

The Graph precision rows pass, but two seed-wise mean errors lie close to the
0.05 boundary; future changes to graph assembly or optimization should retain
this benchmark as a regression test. The experiment does not compare
real-data predictive families.

The formal phase gate passes. A controlled dielectric factorial may now be
run using one frozen backbone/mean, identical splits and budgets, and
family/law changes only. Its results must not be used to retune this recovery
gate, and this report alone does not authorize manuscript result changes.

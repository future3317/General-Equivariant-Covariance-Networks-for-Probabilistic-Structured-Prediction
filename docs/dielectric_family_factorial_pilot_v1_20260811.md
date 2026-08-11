# Dielectric family factorial pilot v1

Date: 2026-08-11

Status: **the declared v1 operational gate failed; no formal factorial was
started from this gate.**

## Existing evidence

The immutable frozen cache contains one common dielectric representation and
mean. Existing E1 artifacts distinguish two Full Student-t controls:

- zero-training evaluation of the cached Full projection: test NLL
  `-2.62468`;
- a fresh centered-e4 Full projection trained for 60 epochs: test NLL
  `-2.72201`.

## New v1 evidence

All eight family/law arms ran from fresh projections for at most 20 epochs on
the complete 4,236/485/281 cached train/validation/test splits. Artifact,
finite, strict-SPD, exact-schema, common-cache, common-target, common-sample-ID,
common-frozen-mean, and validation-only selection checks passed.

The v1 gate nevertheless required the *fresh 20-epoch* Full Student-t arm to
reproduce the *zero-training cached-projection* NLL `-2.6247` within 0.20. Its
test NLL was `0.94490`, so the gate failed. The formal three-seed run was not
launched.

The v1 arm results are development diagnostics only:

| Family | Law | Test NLL | Validation NLL | Selected epoch | Energy Score |
|---|---|---:|---:|---:|---:|
| Isotropic | Gaussian | 3.9666 | 3.8581 | 20 | 0.6188 |
| Isotropic | Student-t | 3.2543 | 3.2423 | 20 | 0.6588 |
| Block | Gaussian | 112.8315 | 0.4021 | 20 | 0.4865 |
| Block | Student-t | -1.3116 | -2.5227 | 20 | 0.4651 |
| Low-rank | Gaussian | 6.2381 | 6.0573 | 20 | 1.3024 |
| Low-rank | Student-t | 5.3903 | 5.2341 | 20 | 1.4079 |
| Full | Gaussian | 6.1712 | 3.4042 | 20 | 0.5446 |
| Full | Student-t | 0.9449 | -0.3034 | 20 | 0.4718 |

Every arm selected the final allowed epoch, so v1 is not a converged matched
family comparison. The large Block-Gaussian validation/test discrepancy also
warns against interpreting these development scores as family rankings.

## Supported inference

The implementation and artifact pipeline are operational on the real frozen
cache. V1 does not support or reject any covariance family or radial law,
because its fresh heads were budget-limited and its reference comparator used
different initialization semantics.

## Protocol correction

V1 remains failed and is not reclassified. A separately versioned v2
operational gate uses the existing zero-training `fixed` path to reproduce the
cached-projection reference within `1e-4`. Fresh factorial heads remain
development-only in that gate; the formal comparison uses the pre-existing
60-epoch matched budget for every arm. This correction follows the historical
E1 protocol record rather than choosing a threshold from v1 arm performance.

Evidence path:
`/home/workspace/lrh/RESULTS/Tpami/dielectric/family_factorial_pilot_36ac9cf`.

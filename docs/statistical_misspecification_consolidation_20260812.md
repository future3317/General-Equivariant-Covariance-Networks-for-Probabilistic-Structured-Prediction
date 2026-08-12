# Statistical misspecification evidence consolidation

Date: 2026-08-12

Status: final evidence consolidation; no training or model exploration was run.

## Main conclusion

The three-seed dielectric confirmation supports conditional ν(x) as a
radial/tail diagnostic repair: it improves proper score, coverage, the radial
PIT statistic, and the whitened second-moment defect under the same frozen
features, mean, scatter family, spectral window, split, and validation-only
selection contract. The radius--direction independence test remains rejected
for every confirmed seed.

The uncertainty-only representation branch is retained only as a negative
diagnostic. Its lower NLL and whitened defect were accompanied by worse Energy,
MACE, Coverage50/90, radial PIT, and no improvement in the scalar alignment
proxy or directional rejection. It is not a formal method and was not promoted
to a full-data or multi-seed study.

## Evidence layers

### Existing evidence

- Fixed-ν=5 dielectric baseline: one frozen test artifact, 281 test samples.
- ITOP fixed/global/conditional-ν and observation-descriptor pilot: Side was
  used for training/validation; Top/OOD was evaluation-only and never used for
  selection. The pilot decision rejects the ITOP repair route for this round.
- Prior mixture-collapse and spectral-window decisions remain unchanged and are
  not re-run here.

### New consolidation evidence

- The three conditional-ν dielectric test artifacts (seeds 42/43/44) were
  re-read and summarized with one shared metric implementation.
- Fixed, conditional-ν seed 42, and representation-repair seed 42 were
  compared from serialized predictions using the same coverage/MACE definition.
- Every input artifact used in the consolidation has a SHA256 entry in
  `artifact_manifest.json`.
- A double-column publication figure is exported as vector PDF and 300 dpi PNG;
  the full metric table is in `dielectric_comparison.csv/json`, and ITOP pilot
  values are preserved in `itop_negative_pilot.json`.

## Dielectric comparison

Values below are test-set quantities in the transformed log--Kelvin--Mandel
coordinate space. NLL is the full normalized Student-t log density. Coverage
uses the law-correct marginal reference quantile; MACE is the mean absolute
error over the predefined 0.1--0.9 and 0.95 levels. The radial PIT KS and
whitened defect are diagnostic statistics, not guarantees of calibration.

| method | seeds | NLL | Energy | Cov50 | Cov90 | Cov95 | MACE | whitened defect | radial PIT KS | max |rho| | perm p | alignment proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed ν=5 | 1 | -2.6247 | 0.4410 | 0.4377 | 0.7117 | 0.7687 | 0.1104 | 10.4619 | 0.2143 | 0.4613 | 0.005 | 0.2635 |
| conditional ν(x) | 3 | -2.8482 ± 0.0002 | 0.4396 ± 0.0020 | 0.4626 | 0.7758 | 0.8577 | 0.0761 ± 0.0002 | 5.6609 ± 0.0060 | 0.1488 ± 0.0001 | 0.4581 | 0.005 | 0.2635 |
| uncertainty-only branch | 1 | -2.9966 | 0.4462 | 0.3523 | 0.7544 | 0.8612 | 0.1190 | 3.3544 | 0.2162 | 0.4559 | 0.005 | 0.2569 |

The conditional-ν row is the mean over seeds 42/43/44; coverage and
permutation resolution are discrete diagnostics and are therefore shown as
the common test value rather than a misleading seed standard deviation. The
representation branch is a single predeclared diagnostic run.

## Interpretation

### Supported inference

Conditional ν(x) is a stable repair of radial/tail misspecification under the
audited frozen representation. It does not establish statistical adequacy of
the full predictive law: direction dependence remains significant at the
reported permutation resolution, and the predeclared uncertainty/error
alignment proxy does not improve.

### Rejected explanation

The one-seed uncertainty-only branch does not support the claim that frozen
representation information is the dominant remaining bottleneck. A lower NLL
alone is insufficient because several calibration and proper-score diagnostics
degraded.

### Unresolved

The remaining error is consistent with non-elliptical directional
misspecification and/or a mismatch in the structured scatter law. These two
causes are not separated by the stopped representation diagnostic. No claim is
made about flow/diffusion, extra mixture components, or a wider spectral
window.

## Reproducibility and scope

Model selection remained validation-only; no test or Top/OOD value entered
selection. Existing checkpoints and predictions were not overwritten. The
machine-readable artifact manifest records source paths, file sizes, SHA256
hashes, split/selection summaries, and the environment label. It is repository
provenance, not manuscript content.

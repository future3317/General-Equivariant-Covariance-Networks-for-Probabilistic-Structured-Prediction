# Dielectric frozen-family factorial results

Date: 2026-08-11

Status: **formal 4-family x 2-law x 3-seed factorial completed and audited.**
The experiment supports Full Student-t as the strongest tested fixed-law
family, while rejecting the claim that any tested single ellipse is calibrated
or statistically correct.

## Existing evidence

E0/E1 had already shown that the released Full Student-t model has both radial
and directional defects. Conditional `nu` improved dielectric test NLL from
`-2.6247` to `-2.9298` and radial KS from `0.2143` to `0.0926`, but did not
repair direction sphericality or radius-direction independence. A matched
spectral-window control excluded the centered-e4 restriction as the principal
cause. The independent NumPy/SciPy recovery experiment subsequently passed all
12 matched family/seed gates, removing the former circular teacher/learner
validation concern.

## Protocol and artifact audit

Every factorial arm used byte-identical cached features, frozen means, targets,
and sample IDs from one source checkpoint. Split sizes were 4,236 train, 485
validation, and 281 test samples. Only the compiler-declared operator family
and radial law changed:

- isotropic: `LowRankCovariance(rank=0)`, one coordinate;
- isotypic block: two coordinates;
- low-rank-plus-isotropic: rank 2, 13 coordinates;
- Full centered-e4: 21 coordinates;
- Gaussian or Student-t with fixed `nu=5`.

Seeds `42,43,44` changed projection initialization and train-loader order.
Every arm used AdamW (`5e-4`, weight decay `1e-5`), at most 60 epochs, patience
5, and validation NLL only for checkpoint selection.

The first 20-epoch pilot remains a failed development artifact because it
incorrectly compared a fresh projection to a zero-training cached-projection
reference. The corrected v2 operational control reproduced the fixed-cache NLL
`-2.62468379` with absolute error `1.62e-5`; it did not reclassify the v1 run.

For the formal run, all 24 arms passed:

- clean source commit
  `0b5ea921a110d2db03c12a77313bb1cbaee4c56e`;
- exact active-family executor and bijective checkpoint mapping;
- requested family/law schema and coordinate count;
- finite predictions and strict FP64 SPD reconstruction;
- validation-only selection;
- all checkpoint, prediction, JSON, and manifest hashes;
- common cache, test IDs, targets, and frozen-mean hashes;
- local re-hashing after download.

Evidence paths:

- server:
  `/home/workspace/lrh/RESULTS/Tpami/dielectric/family_factorial_formal_0b5ea92`;
- local: `results/dielectric_family_factorial_0b5ea92`;
- failed pilot record:
  `docs/archive/2026-08-18/dielectric_family_factorial_pilot_v1_20260811.md`.

## New formal evidence

Mean and sample standard deviation across the three training seeds are:

| Family | Law | Test NLL | Energy Score |
|---|---|---:|---:|
| Isotropic | Gaussian | 1.3392 +/- 0.3817 | 0.4575 +/- 0.0051 |
| Isotropic | Student-t | 0.2528 +/- 0.1359 | 0.4545 +/- 0.0051 |
| Block | Gaussian | 5.1695 +/- 8.9999 | 0.4500 +/- 0.0129 |
| Block | Student-t | -0.5366 +/- 1.6926 | 0.4685 +/- 0.0280 |
| Low-rank | Gaussian | 5.7668 +/- 1.4575 | 0.7446 +/- 0.0191 |
| Low-rank | Student-t | 4.2772 +/- 0.1345 | 0.8018 +/- 0.0209 |
| Full | Gaussian | -0.1811 +/- 1.2617 | 0.4510 +/- 0.0053 |
| **Full** | **Student-t** | **-2.3156 +/- 0.2448** | **0.4479 +/- 0.0067** |

Student-t minus Gaussian paired NLL deltas were negative for all seeds in the
Full (`-3.424,-1.504,-1.475`), isotropic
(`-1.370,-0.962,-0.927`), and low-rank
(`-0.747,-3.017,-0.704`) families. Block was unstable
(`-17.168,+1.508,-1.458`): one Gaussian seed had test NLL `15.56`, and one
Student-t seed selected epoch 4 with test NLL `1.41`. These finite artifacts
pass the engineering gate but do not support a stable Block ranking.

Full Student-t had the lowest test NLL for every seed and lower Energy Score
than each other Student-t family for every seed. Relative to Full Gaussian,
Full Student-t also improved Energy Score in all three seeds, although the
absolute changes were small.

### Calibration and falsification diagnostics

| Family | Law | Cov90 | Cov95 | Radial KS | Projection median KS | Direction defect |
|---|---|---:|---:|---:|---:|---:|
| Isotropic | Gaussian | 0.822 | 0.852 | 0.325 | 0.139 | 3.387 |
| Isotropic | Student-t | 0.716 | 0.790 | 0.215 | 0.140 | 3.387 |
| Block | Gaussian | 0.823 | 0.842 | 0.399 | 0.154 | 2.875 |
| Block | Student-t | 0.578 | 0.619 | 0.342 | 0.168 | 3.108 |
| Low-rank | Gaussian | 0.985 | 0.985 | 0.865 | 0.312 | 3.065 |
| Low-rank | Student-t | 0.986 | 0.989 | 0.840 | 0.318 | 3.051 |
| Full | Gaussian | 0.848 | 0.867 | 0.435 | 0.163 | 2.882 |
| Full | Student-t | 0.677 | 0.753 | 0.240 | 0.141 | 2.868 |

Values are seed means and use the correct law-specific ellipsoidal thresholds.
For Full, Student-t improved radial and projection KS in every seed, consistent
with a real radial-law effect. It did not yield nominal coverage. All 24 arms
rejected radius-direction independence at permutation `p <= 0.02`; nearly all
were at the minimum reported `p=0.005`. Direction defects also remained large.
Thus better proper NLL is not evidence of a correct or calibrated predictive
distribution.

## Supported inference

Two independent modeling axes matter on frozen dielectric features:

1. **Radial law:** fixed-`nu` Student-t is consistently preferable to Gaussian
   for Full and isotropic operators under proper NLL, with synchronized but
   modest Energy/projection improvements.
2. **Operator family:** Full Student-t is consistently better than isotropic,
   rank-2, and Block Student-t across seeds. Restricted covariance coordinates
   are not an adequate replacement for Full on this frozen representation.

This is direct real-data evidence for the compiler's separation of operator
family and radial-law semantics. It is stronger than comparing another Full
SPD parameterization because the backbone, mean, samples, budget, and selection
rule are fixed.

## Rejected explanations

- The dielectric failure is not primarily caused by using too many Full
  coordinates: isotropic, Block, and rank-2 heads do not improve the matched
  proper-score result.
- Gaussian tails are not adequate under the matched Full protocol.
- Better Full Student-t NLL does not establish calibration: Coverage90/95,
  direction, and radius-direction tests remain decisively wrong.
- Block is not a robust second-best family; its seed variance is too large.

## Unresolved and phase decision

The fixed-`nu` Full Student-t factorial mean (`-2.3156`) does not surpass the
released fixed projection (`-2.6247`) or E1 conditional-`nu` result
(`-2.9298`). This is expected to remain a controlled family/law ablation rather
than replace the released checkpoint. Conditional `nu` still addresses only
the radial axis, while direction/orientation misspecification remains.

The P0 dielectric family comparison is complete. No further isotropic/Block/LR
sweep, spectral-window expansion, mixture, temperature, or conformal repair is
justified by this result. The dataset has no repeated labels, so none of these
scatters is identified as physical/DFPT aleatoric covariance. Manuscript
integration should present the factorial as controlled compiler-family evidence
and retain the explicit misspecification limitation; it is not performed in
this phase.

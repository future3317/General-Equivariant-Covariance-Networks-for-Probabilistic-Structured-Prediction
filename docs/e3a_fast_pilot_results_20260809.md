# E3a fast-pilot result and phase-gate decision

Date: 2026-08-09

Status: **development-only falsification evidence**. This is not the final
512-point, full-data, three-seed E3 result and must not be inserted into the
manuscript main tables.

## Question and protocol

E3 asks whether diversity from independently learned mean functions explains
the predictive-distribution failures left after E0--E2. The pilot deliberately
used the cheapest protocol that could reject an unpromising route:

- 2,487 Side training frames and 256 points;
- model seeds 42 and 43, with shared `split_seed=42`;
- independently initialized deterministic backbone and mean for each seed;
- each seed's own frozen Full Student-t head with fixed `nu=5`;
- validation-only checkpoint selection; Side test is IID and Top is a
  predeclared cross-view OOD evaluation;
- exact equal-weight finite-mixture `logsumexp` density, with no moment-matched
  pseudo-density.

Training members were produced from clean commit `c40898f`; the freeze-contract
repair is commit `9611882`; the strict two-member evaluator is commit
`ea73ba2`. The server result is
`/home/workspace/lrh/RESULTS/Tpami/E3/pilot_c40898f/itop_development_n256` and
the local mirror is `results/e3_pilot_c40898f/itop_development_n256`.

## Artifact and provenance audit

- Both deterministic members pass the predeclared Side MPJPE threshold of
  35 cm: 32.4567 cm and 32.7724 cm.
- The two deterministic checkpoints, frozen feature caches, and Full-t heads
  are distinct across seeds. Neither seed consumes the other seed's checkpoint
  or feature cache.
- The frozen evaluator records clean source, fixed split/cache/compiler
  contracts, member checkpoint chains, upstream deterministic checkpoint
  hashes, and prediction hashes.
- Side and Top each contain exactly 4,863 samples. All tensors in 14 downloaded
  prediction files are finite.
- All 38 locally recomputed artifact hashes match the saved provenance.
- Every Full-t test artifact passed strict FP64 scale materialization while the
  generator/operator algebra remained FP32.
- The ensemble density is recorded as
  `equal_weight_exact_finite_student_t_logsumexp`; projection PIT uses the
  weighted component Student-t CDF and is explicitly not moment matched.
- Top was never used for member or ensemble selection.

The obsolete monitor is paused, no TPAMI training/evaluation process remains,
and server GPU 2 is idle. The partial seed-43 joint log and the original failed
preflight directory are retained as negative engineering evidence, not treated
as completed members.

## New E3 pilot evidence

### Independent frozen members and their exact mixture

The Energy Scores in this table are recomputed by the common ensemble evaluator
so that members and the mixture use the same sampling protocol.

| Split | Predictor | MPJPE (cm) | Exact NLL | Energy Score (m) |
|---|---|---:|---:|---:|
| Side IID | seed 42 | 32.4567 | -38.3760 | 1.0550 |
| Side IID | seed 43 | 32.7727 | -44.4059 | 1.0192 |
| Side IID | exact 2-member mixture | **31.5882** | **-44.8838** | **1.0130** |
| Top OOD | seed 42 | 67.1426 | 34.3105 | 2.0718 |
| Top OOD | seed 43 | 68.1016 | **14.8059** | 2.0474 |
| Top OOD | exact 2-member mixture | **65.5943** | 15.3733 | **1.9738** |

The Side exact density improves over the better member by 0.4779 NLL and also
slightly improves Energy Score and the ensemble-mean point estimate. However,
all 64 Side projection-PIT directions still reject after Bonferroni correction
(median KS 0.1638). On Top, the mixture improves point error and Energy Score,
but its exact NLL is 0.5674 worse than the best member and all 64 projection
directions reject.

Between-member spread is large enough to detect the view shift but does not
track which examples are wrong:

| Diagnostic | Side IID | Top OOD |
|---|---:|---:|
| frame error vs spread Spearman | 0.0508 | -0.1413 |
| joint error vs spread Spearman | 0.1850 | 0.0645 |
| mean between-member trace (m2) | 0.1186 | 0.5131 |

The Top/Side mean-trace ratio is 4.3277 and the Side-to-Top spread AUROC is
0.9573. These are shift diagnostics only; they are not calibration evidence.
With only two members, leave-one-out is identical to the two individual-member
rows and no three-member robustness claim is possible.

### Joint-training falsification

The limited Side benefit was sufficient to run the predeclared maximum
three-epoch joint pilot on seed 42. It revealed a monotone NLL--mean coupling
pathology rather than a useful end-to-end uncertainty model.

| Validation stage | LR | Epoch | NLL | MPJPE (cm) | Mean gradient norm |
|---|---:|---:|---:|---:|---:|
| frozen baseline | 5e-4 | 5 | -52.8814 | **30.2100** | 23.20 |
| joint | 5e-4 | 1 | -75.1749 | 41.4261 | 125.89 |
| joint | 5e-4 | 2 | -81.4147 | 45.3890 | 224.83 |
| joint | 5e-4 | 3 | **-85.3556** | 47.3761 | 314.26 |
| joint low-LR control | 5e-5 | 1 | -60.5502 | 32.2999 | 68.80 |
| joint low-LR control | 5e-5 | 2 | -64.2861 | 33.8430 | 70.21 |
| joint low-LR control | 5e-5 | 3 | -67.3030 | 35.6135 | 81.65 |

The default joint checkpoint reaches Side test NLL -59.5427 but degrades
MPJPE to 45.9371 cm and Energy Score to 1.4444 m. Reducing only the learning
rate by ten times slows but does not remove the tradeoff: Side test NLL is
-49.9841, MPJPE is 35.7811 cm, and Energy Score is 1.1612 m, compared with the
same seed's frozen values of -38.3760, 32.4567 cm, and 1.0926 m. The low-LR
control also worsens Top NLL to 20.8592. Seed 43 joint training was stopped
after the predeclared failure became decisive; its partial log is not a model
artifact.

## Existing evidence

- E0 rejects the single fixed-nu elliptical law on ITOP and dielectric.
- E1 isolates a dielectric radial-tail issue but rejects dielectric K=2 and an
  SPD spectral-window explanation. Its ITOP K=2 gain remained mechanistically
  ambiguous.
- E2 identifies the ITOP K=2 gain mainly as a frozen-mean bias/density-mass
  surrogate, rejects the tested low-capacity observation-information route,
  and leaves model/function uncertainty as the final live E3 hypothesis.

## Supported inference

Independent means provide a small Side IID model-averaging benefit in point
prediction, exact density, and Energy Score. This supports diversity in mean or
mode coverage. It does **not** support a useful epistemic uncertainty estimator:
between-member spread has nearly no positive relation to held-out frame error,
and the best-member Top density is better than the ensemble density.

The joint experiment provides direct evidence that the current unconstrained
heteroscedastic NLL can improve by moving the mean into a substantially worse
point-estimation regime. The remaining actionable root cause is therefore the
mean/scatter training objective and model misspecification, not another SPD
parameterization.

## Rejected explanations and stopped work

The strict E3 gate is not met. Do not launch the full 512-point, three-seed
confirmation, E3b bootstrap/subsample members, a dielectric deep ensemble,
learned ensemble weights, a public finite-mixture primitive, or more joint
learning-rate sweeps. More members cannot repair the demonstrated absence of
error-aligned spread in this pilot without a new hypothesis.

## Unresolved

- A two-member development panel cannot estimate final-seed robustness.
- The source of the independently learned mean difference is not identified as
  posterior uncertainty; it may be optimization variability or model bias.
- The current experiment does not test a training objective that preserves the
  deterministic mean while fitting scatter.
- Repeated dielectric protocol/condition labels remain unavailable, so
  physical or protocol-level aleatoric covariance remains unidentifiable.

The next admissible experiment, if pursued, is a one-seed development
falsification of the objective-coupling hypothesis: preserve an explicit MSE
mean objective and prevent the scatter NLL term from updating the mean path,
without changing the Student-t family or SPD compiler. It requires a separate
protocol and gate; it is not an automatic continuation of E3.

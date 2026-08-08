# E1 frozen-distribution result audit

Date: 2026-08-08

Scope: E1 only. No E2/E3 experiment was started and no manuscript result was
changed. The intervention freezes one checkpoint-derived feature representation
`H`, mean `mu`, and (for the distribution-law comparison) scatter parameters.
All selection uses the declared validation split. ITOP Top is OOD-only.

## Controls and artifact validity

- Dielectric uses the released unified Full-coordinate Student-t checkpoint
  (`3864f5cf...`) and one immutable cache (`218159ac...`): 4,236 train, 485
  validation, and 281 test structures.
- ITOP uses the released frozen Full-t checkpoint (`76038afb...`) and one
  immutable cache (`b518fb24...`): 16,192 Side train, 1,799 Side validation,
  4,863 Side test, and 4,863 Top OOD poses.
- Mean, features, split IDs, optimizer, FP32 operator/NLL algebra, patience 5,
  and validation-NLL model selection are fixed. Conditional-nu trains only one
  invariant `0e` readout. K=2 trains only an equivariant offset; weights are
  `(1/2,1/2)`, component nu is 5, and scatter is shared and frozen.
- Twelve completed runs passed manifest SHA checks, finite-prediction checks,
  split-count checks, and source-provenance checks. Exact single Student-t and
  exact finite-mixture `logsumexp` semantics are recorded in every protocol.
- The generic composition is certified structurally: `delta` is an O(3)
  equivariant linear readout, so `mu +/- delta` is equivariant; fixed scalar
  weights are invariant; the shared scatter uses the existing compiled SPD
  operator and its equivariance/SPD certificate. The numerical/unit checks are
  in `tests/test_frozen_distribution_readout.py` and the compilation certificate
  is embedded in each `protocol.json`.

Server evidence roots:

- `/home/workspace/lrh/RESULTS/Tpami/E1/852c274`
- `/home/workspace/lrh/RESULTS/Tpami/E1/bd3e73a`
- `/home/workspace/lrh/RESULTS/Tpami/E1/bde59d1`

Local evidence mirror:
`results/e1_frozen_distribution_20260808`. Core artifacts for all runs and
canonical held-out prediction tensors are mirrored locally and their hashes
match the server manifests. Repeated train predictions remain on the server.

## Existing evidence entering E1

| Evidence | State before E1 | Path |
|---|---|---|
| Fixed-nu single ellipse rejected | Verified for dielectric Full-t and ITOP Full/Graph-t | `results/uncertainty_root_cause_d945901` |
| Student-t density, scale semantics, SPD assembly | Implemented and tested | `distributions/student_t.py`, compiled operator programs |
| Conditional nu | Missing | No prior artifact |
| Trained K=2 predictor | Missing | No prior artifact |
| Exact finite-mixture NLL | Evaluator existed, no trained mixture evidence | `evaluation/ensemble.py` |
| Strict matched spectral control | Missing | Old checkpoints used unmatched protocols |

## New E1 evidence: dielectric Full coordinates

Held-out test metrics (seed 42):

| Distribution | Selected epoch | Exact NLL | Energy Score | Radial PIT KS | Projection median KS | Projection decile L1 | Direction defect |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed nu=5 | 0 | -2.6247 | 0.4410 | 0.2143 | 0.1478 | 0.2343 | 2.8931 |
| Conditional nu | 60 | **-2.9298** | **0.4403** | **0.0926** | 0.1331 | **0.1359** | 2.8931 |
| K=2 symmetric mixture | 55 | -2.7100 | 0.4415 | n/a | **0.1277** | 0.2063 | n/a |

Conditional nu predicts test values in `[2.053, 2.885]` (median 2.199). It
substantially improves radial fit and NLL, while the direction defect is exactly
unchanged and radius-direction independence remains rejected (`p=0.005`). This
is positive evidence for radial heterogeneity, not a repair of the elliptical
direction law.

The dielectric K=2 intervention improves NLL and projection PIT modestly but
worsens Energy Score. It therefore does not meet the stable multi-score gate and
must not be expanded to learned gates or component-specific scatters on this
task.

## New E1 evidence: matched dielectric spectral control

All rows retrain the same typed scatter projection from the same source
initialization; only the SPD map changes.

| SPD map | Selected epoch | Test NLL | Energy Score | Projection decile L1 | Max condition number | Status |
|---|---:|---:|---:|---:|---:|---|
| Centered `[-2,2]` (`e4`) | 60 | **-2.7220** | **0.4439** | **0.2408** | 54.60 | valid |
| Centered `[-4,4]` (`e8`) | 60 | -2.6765 | 0.4503 | 0.2946 | 2,982.31 | valid |
| Unbounded matrix exponential | 60 | not accepted | not accepted | not accepted | n/a | numerical SPD gate failed |

The unbounded run monotonically improved validation NLL through epoch 60, but
the selected model produced a test maximum scale eigenvalue of
`1.705e12` and one FP32 non-positive minimum eigenvalue (`-2.909e3` from
eigendecomposition roundoff at the extreme dynamic range). It is an invalid
prediction artifact, not a scored baseline; no jitter or fallback was applied.
The wider valid window is worse than `e4` on held-out NLL, Energy Score, and
projection calibration. Spectral restriction is therefore not supported as the
principal causal explanation.

## New E1 evidence: ITOP Full-t

The baseline NLL reproduces the released artifact within `2e-6`. The table
reports mean +/- sample standard deviation over paired diagnostic/training
seeds 42/43/44 for fixed and K=2. Conditional nu is the preregistered seed-42
run. All three K=2 runs use the same frozen checkpoint and fixed split.

### Side IID test (4,863 poses)

| Distribution | Exact NLL | Energy Score | Projection median KS | Projection decile L1 |
|---|---:|---:|---:|---:|
| Fixed nu=5 | -70.8909 +/- 0.0000 | 0.70854 +/- 0.00020 | 0.1176 +/- 0.0057 | 0.2082 +/- 0.0065 |
| Conditional nu | -70.7559 | 0.71001 | 0.1307 | 0.2508 |
| K=2 symmetric mixture | **-73.2000 +/- 0.0259** | 0.70848 +/- 0.00033 | **0.0942 +/- 0.0122** | **0.1245 +/- 0.0182** |

Paired K=2-minus-fixed changes are `-2.3090 +/- 0.0259` NLL,
`-0.00007 +/- 0.00025` Energy Score, and `-0.0838 +/- 0.0194` projection
decile L1. Thus exact held-out NLL and mixture-aware projection calibration
improve in all three seeds; Energy Score is effectively neutral and changes
sign in one of three seeds. Projection rejection is not eliminated.

### Top cross-view OOD (4,863 poses; never used for selection)

| Distribution | Exact NLL | Energy Score | Projection median KS | Projection decile L1 |
|---|---:|---:|---:|---:|
| Fixed nu=5 | 9.6335 +/- 0.0000 | 2.48437 +/- 0.00036 | 0.6804 +/- 0.0502 | 1.2647 +/- 0.0425 |
| Conditional nu | 12.5391 | 2.50433 | 0.6816 | 1.2812 |
| K=2 symmetric mixture | **9.4675 +/- 0.0193** | **2.48319 +/- 0.00049** | 0.6569 +/- 0.0641 | **1.0306 +/- 0.0883** |

K=2 is directionally better OOD, but all 64 projection tests still reject and
the absolute OOD defect remains severe. This is not evidence of cross-view
calibration.

## Phase-gate answers

1. **Does conditional nu improve radial fit but not directional mismatch?**
   Yes for dielectric: radial/NLL evidence improves while direction statistics
   are unchanged. No for ITOP: the learned nu is larger and held-out NLL,
   Energy Score, radial PIT, and projection calibration all worsen. Conditional
   radial heterogeneity is task-dependent, not a universal root cause.
2. **Does K=2 stably improve held-out proper NLL and mixture-aware
   calibration/score?** Yes for ITOP exact NLL and projection PIT over three
   offset-training seeds; Energy Score is neutral on Side and slightly better
   on Top. No for dielectric under the minimal shared-scatter intervention.
   This is sufficient to permit design review of a generic finite-mixture
   compiler composition, but not to claim general efficacy or to add learned
   gates/component scatters without another controlled gate.
3. **Is the dielectric centered spectral restriction the principal cause?**
   No. Widening the matched window worsens every held-out score, and the
   unbounded FP32 run fails numerical SPD validity under the same protocol.

## Inference versus unresolved questions

Supported inference:

- Dielectric failure contains a strong radial-tail component; conditional nu
  addresses it without repairing the rejected directional law.
- ITOP frozen features contain enough signal for a symmetric location offset to
  improve exact mixture NLL and projection PIT on Side. This does not by itself
  prove physical multimodality: the offset may absorb structured mean bias.
- The `e4` centered window is not the main dielectric bottleneck under the
  matched control.

Still unresolved:

- Whether the ITOP gain is true conditional multimodality, residual mean bias,
  or a proxy for missing observation information.
- Why ITOP Energy Score changes negligibly despite a large NLL gain.
- Whether a general mixture primitive remains beneficial beyond this one
  frozen ITOP Full-t checkpoint and output family.
- Information sufficiency of ITOP `H` versus raw observation descriptors (E2).
- End-to-end model uncertainty (E3). Neither E2 nor E3 was started here.

## Engineering changes used by E1

- `852c274`: exact Energy Score all-pairs calculation made memory-bounded by
  chunking; float64 values and sample gradients match the dense estimator.
- `bd3e73a`: conditional readout preserves frozen parameter coordinates in the
  task-neutral prediction schema.
- `bde59d1`: reproducible ITOP Full-t adapter into the same frozen-distribution
  cache schema; no ITOP-specific trainer or probability branch was added.


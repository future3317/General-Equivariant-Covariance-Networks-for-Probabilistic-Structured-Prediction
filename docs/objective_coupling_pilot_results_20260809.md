# ITOP objective-coupling pilot results

Date: 2026-08-09

Status: **positive two-seed development evidence**. The runs use 2,487 Side
training frames and 256 points. They are not final full-data results and the
manuscript main tables remain unchanged.

## Existing evidence

The E3 fast pilot showed that ordinary joint Student-t NLL training reduced NLL
by degrading the mean. At LR 5e-4, seed-42 validation MPJPE rose from 30.2100
to 47.3761 cm while NLL improved from -52.8814 to -85.3556. A ten-times-lower
LR slowed but did not remove the tradeoff. The repository already contained a
generic faithful objective used by dielectric: MSE trains the mean path while
the scatter NLL uses a detached mean residual and detached features.

## Exact intervention

The only new wiring exposes that existing faithful objective through the
controlled ITOP readout and trainer. Each seed starts from its own selected
frozen Full Student-t checkpoint. All data, split, architecture, fixed `nu=5`,
Full SPD compiler program, optimizer, LR 5e-4, FP32 operator/NLL algebra,
three-epoch limit, and validation-NLL selection are unchanged.

Under the faithful boundary:

- backbone and direct mean receive MSE gradients only;
- covariance projection receives detached-feature residual-NLL gradients;
- the compiled operator lifting is reused from the frozen head and receives no
  gradient during this stage;
- evaluation uses the ordinary proper Student-t density, not the composite
  training objective.

## Artifact audit

| Seed | Source commit | Frozen checkpoint SHA prefix | Side/Top samples | Hash/finite result |
|---:|---|---|---:|---|
| 42 | `c6a293c` | `1399763745b0` | 4,863 / 4,863 | 6/6 hashes match; all tensors finite |
| 43 | `7493676` | `3a843366ea41` | 4,863 / 4,863 | 6/6 hashes match; all tensors finite |

Both source worktrees were clean, use the same dataset-cache hash
`d0fe3fc1...1127`, and share `split_seed=42`. Each `args.json` records
`faithful_joint=true`; together with the clean source commit this identifies
the implemented `mse_mean_plus_detached_feature_residual_nll` semantics. The
pilot artifacts predate the later human-readable `gradient_routing` field, so
their `freeze.boundary` string is still the generic joint-stage description;
the source, flag, checkpoint hash, and training-contract hash are the operative
evidence. The current contract patch makes this boundary explicit for future
runs without changing the objective. Top never selected a checkpoint. Server
and local evidence paths are:

- `/home/workspace/lrh/RESULTS/Tpami/E3/pilot_c40898f/itop_development_n256/seed_42/joint_full_student_t_faithful_c6a293c`
- `/home/workspace/lrh/RESULTS/Tpami/E3/pilot_c40898f/itop_development_n256/seed_43/joint_full_student_t_faithful_7493676`
- `results/e3_pilot_c40898f/itop_development_n256/seed_42/joint_full_student_t_faithful_c6a293c`
- `results/e3_pilot_c40898f/itop_development_n256/seed_43/joint_full_student_t_faithful_7493676`

## New development evidence

### Validation trajectories

| Seed | Epoch | NLL | MPJPE (cm) | Mean gradient norm |
|---:|---:|---:|---:|---:|
| 42 | 1 | -56.4327 | 29.3468 | 17.96 |
| 42 | 2 | -58.3440 | 29.0730 | 18.68 |
| 42 | 3 | **-60.2514** | **28.6364** | 20.17 |
| 43 | 1 | -59.3377 | 30.4524 | 18.22 |
| 43 | 2 | -59.6603 | **29.5908** | 18.26 |
| 43 | 3 | **-61.0568** | 29.8573 | 18.03 |

Unlike ordinary joint training, neither seed shows monotone mean collapse or
exploding gradients while validation NLL improves.

### Held-out comparison against each seed's frozen checkpoint

| Split | Seed | Stage | MPJPE (cm) | NLL | Energy Score (m) |
|---|---:|---|---:|---:|---:|
| Side IID | 42 | frozen | 32.4567 | -38.3760 | 1.0926 |
| Side IID | 42 | faithful | **31.4449** | **-44.1415** | **1.0287** |
| Side IID | 43 | frozen | 32.7727 | -44.4059 | 1.0510 |
| Side IID | 43 | faithful | **31.5313** | **-46.7687** | **1.0113** |
| Top OOD | 42 | frozen | 67.1426 | 34.3105 | 2.1381 |
| Top OOD | 42 | faithful | **66.7723** | **26.8137** | **2.0822** |
| Top OOD | 43 | frozen | **68.1016** | **14.8059** | **2.0890** |
| Top OOD | 43 | faithful | 69.9636 | 21.5680 | 2.1723 |

Both seeds pass the predeclared Side gate: proper NLL and Energy Score improve
without point-estimation degradation. Seed 42 also improves every reported Top
score, while seed 43 worsens every reported Top score. The intervention is
therefore stable for Side IID but not for cross-view OOD.

## Supported inference

The ordinary joint-NLL failure was substantially caused by gradient coupling,
not by an unavoidable conflict between point prediction and Full Student-t
density. Separating mean MSE gradients from scatter NLL gradients is an
actionable training correction and is more promising than adding another SPD
family, mixture gate, or learning-rate sweep.

## Rejected explanation

This does not validate epistemic uncertainty, physical aleatoric covariance,
or Top calibration. The opposite seed-43 Top response rejects any claim that
the faithful boundary alone solves cross-view robustness. It also does not
revive the stopped E3 ensemble, public mixture, or observation-aware branches.

## Unresolved and next gate

The positive result needs a formal full-data, 512-point comparison before it
can alter manuscript main results. That confirmation should compare frozen
Full-t, ordinary joint, and faithful joint from the same deterministic seed(s),
with Side validation-only selection and Top OOD evaluation. It should not be
launched automatically: first define the exact seed count and compute budget,
and reuse the existing deterministic/frozen checkpoints whenever their data and
geometry contracts match.

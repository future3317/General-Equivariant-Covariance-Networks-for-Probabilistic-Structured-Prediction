# Feedback execution record

This record captures the first evidence-driven pass of the feedback plan. The
experiment was run on the server in the `equivcompiler` environment using the
existing single-seed dielectric Student-t checkpoint
`final_student_t_h64_window2`.

## Diagnostic result

| quantity | held-out value |
|---|---:|
| Mahalanobis² mean (Student-t scale) | 19.8701 |
| symmetric-whitened angular defect | 8.3916 |
| radial PIT mean | 0.5146 |
| radial PIT standard deviation | 0.3647 |
| irrep defect, `0+` | 2.4241 |
| irrep defect, `2+` | 4.4495 |
| irrep defect, `4+` | 6.6891 |
| covariance operator basis dimension | 21 |
| covariance projection local Jacobian rank | 21 |

The full covariance head is locally full rank. The large angular defect,
especially in the `4+` component, supports an orientation/shape mismatch
hypothesis; it is not evidence that the existing head lacks the 21-dimensional
operator basis. The radial PIT is not enough to explain the defect by itself.

## Implemented diagnostics and primitives

- symmetric inverse-square-root whitening;
- Student-t radial PIT and covariance convention tests;
- basis-independent angular whitening defect;
- irrep-resolved whitening defect;
- Gaussian/Student-t Energy Score and isotropic sliced CRPS;
- centered log-volume/log-shape spectral SPD map;
- automatic `Lambda^2(V)` decomposition and skew operator basis;
- zero-initialized equivariant isospectral orientation calibrator;
- strict three-stage mean/covariance/joint protocol with an immutable RunSpec;
  covariance fitting freezes every parameter except the compiled covariance
  projection, rather than relying on feature-detach flags.

## Verification

The latest server regression suite passes with `247 passed, 12 skipped` in
84.90 seconds.  The skipped cases are optional backend/data tests; no test
failure remains.  A two-epoch, 64-example single-seed smoke run also completed
on one CUDA device with the centered spectral-window Student-t configuration:
validation loss `11.0114`, validation physical MAE `5.4639`, test loss
`11.6100`, and test physical MAE `5.9723`.  A subsequent evaluate-only run
reconstructed the same parameterization, distribution, covariance-detach, and
frozen-mean settings and reproduced the metrics without dtype or configuration
errors.

The current final centered-window run predates the unified three-stage
protocol. It is retained as diagnostic evidence only and must not be presented
as the final-method result. New runs write `run_spec.json`, which is the sole
semantic model contract for training, evaluation, calibration, and figures;
`args.json` is retained as operational/data-loader metadata and is never used
to construct a model.

## Unified staged-protocol server smoke test

The server executed `mean -> covariance -> joint` on a 64-example training
subset for one epoch per stage.  The run directories are
`RESULTS/Tpami/dielectric/staged_protocol_smoke/{mean,covariance,joint}`.
Each stage wrote `run_spec.json`, `stage.json`, `best_model.pt`, and final
validation/test diagnostics.  The covariance stage exposed exactly 13
trainable parameters (the compiled covariance projection) and froze the
remaining model; mean and joint exposed 18,734 parameters.  An independent
`--evaluate_only` invocation of the joint checkpoint reconstructed it from
`run_spec.json` and reproduced validation NLL `12.6026` and test NLL `13.0698`.
These smoke metrics are deliberately not scientific results.

## Synthetic causal ablation

On the server, a teacher EIOC applied a known orientation corruption to a fixed
six-dimensional spectrum.  A zero-initialized student EIOC was fitted for 600
steps on 512 samples.  Gaussian NLL changed from `1.85661` to `1.81691`, while
the maximum eigenvalue, log-determinant, and condition-number discrepancies
were `2.3e-6`, `1.9e-6`, and `1.1e-5`, respectively.  This is a causal
orientation test, not a dielectric performance claim: it verifies that the
module can improve the proper score without using spectral rescaling.

The final staged dielectric run and quantitative ITOP benchmark remain
separate experiments and are not represented by this synthetic result.

## Paper build

The TPAMI source was previously rebuilt with BibTeX and LaTeX into
`output/pdf/bare_jrnl_new_sample4.pdf`.  That build predates the final unified
run and is superseded by the PDF produced in the current revision below.

## Superseded centered-window Student-t run (provenance only)

The full single-seed run used one RTX 4090, batch size 128, BF16 backbone
autocast, two data-loader workers, centered spectral-window covariance, and
Student-t likelihood with nu=5. Early stopping selected the best checkpoint at
epoch 69 and stopped at epoch 84. The held-out results were:

| split | NLL | physical MAE | log-KM MAE |
|---|---:|---:|---:|
| validation | -3.5150 | 1.5902 | 0.1304 |
| test | -2.2738 | 1.9941 | 0.1531 |

The final test audit reports mean Mahalanobis squared distance `30.351`, 90%
ellipsoid coverage `0.6655`, 95% coverage `0.7260`, mean condition number
`20.80`, and maximum condition number `54.598`, matching the certified centered
shape-window bound. The symmetric-whitened angular defect is `12.440`; irrep
defects are `5.012` (`0+`), `6.643` (`2+`), and `9.246` (`4+`). The covariance
projection has operator dimension 21 and local Jacobian rank 21, so the
calibration failure is not caused by a missing full-scatter degree of freedom.

Validation-only scalar temperature fitting returned `T=0.6334`; applying it
to the test split worsened test NLL from `-2.2659` to `-2.0268`, because the
test distribution is more dispersed than validation. Block temperatures
(`1.4611` for `0+`, `0.4064` for `2+`) likewise worsened test NLL to `-1.9254`.
These are held-out transfer diagnostics, not post-hoc test fitting, and show
that one validation temperature is insufficient for this shift.

The first audit pass exposed and fixed an evaluation-only reconstruction bug:
`centered_spectral_window` was being rebuilt as the old full-covariance map by
the figure/calibration loader. All final numbers above come from the corrected
loader. The corrected server regression is `247 passed, 12 skipped`; regenerated
diagnostic figures contain no warning.

Additional proper-score diagnostics on the same test split are Energy Score
`0.4256`, isotropic sliced CRPS `0.1444`, and uncertainty-ranking
risk-coverage AUC `0.5523` in compiled-irrep coordinates. These scores are
stored with the other raw JSON artifacts under
`experiments/results/final_student_t_centered_b128/`.

## Unified final single-seed dielectric run

The final TPAMI protocol was run on one RTX 4090 with the same centered
spectral-window Student-t (`nu=5`) contract throughout all three stages. The
mean stage early-stopped at epoch 46, covariance fitting at epoch 76, and
joint fine-tuning used `lr=1e-4`.

| stage / split | NLL | physical MAE | log-KM MAE |
|---|---:|---:|---:|
| mean / validation | not selected | 1.5975 | 0.1336 |
| mean / test | not selected | 2.0616 | 0.1585 |
| covariance / validation | -3.2933 | 1.5975 | 0.1336 |
| covariance / test | -2.1115 | 2.0615 | 0.1585 |
| joint / validation | -3.7151 | 1.6486 | 0.1363 |
| joint / test | **-2.6249** | **2.0611** | **0.1604** |

The final joint test probabilistic diagnostics are Energy Score `0.4426`,
isotropic sliced CRPS `0.1551`, risk-coverage AUC `0.5628`, and mean
Mahalanobis squared distance `25.3222`. Ellipsoid coverage is `43.8%`,
`59.4%`, `71.2%`, and `76.9%` at nominal 50%, 80%, 90%, and 95% levels.
The mean condition number is `26.76`, with maximum `54.5975`, matching the
centered shape-window bound `exp(4)` up to numerical precision.

Validation-only scalar temperature (`T=0.6108`) worsens held-out test NLL from
`-2.6230` to `-2.4247`; block temperature is also worse (`-2.3849`). Neither
calibration transform should be applied to the reported final checkpoint.

The final artifacts are stored under
`experiments/results/unified_student_t_centered_b128_20260726/joint/` and the
server run directory. These are single-seed results and should be presented
with that limitation; the substantial angular whitening defect (`10.55`) and
undercoverage remain open method limitations rather than being hidden by
post-hoc temperature fitting.

## Current TPAMI document build

The revised source is
`E:/PAPER/General Equivariant Covariance Networks for Probabilistic Structured Prediction/bare_jrnl_new_sample4.tex`.
It compiles with the TPAMI IEEEtran template to
`output/pdf/bare_jrnl_new_sample4.pdf` (22 pages). The final Student-t figures
are copied into `figures/dielectric_final/`; the spectral figure was regenerated
with the centered-shape condition bound `exp(4)`, rather than the loose
coordinate-wise volume bound. The rendered pages 15--17 and 21--22 were
visually checked; no clipping, overlap, black blocks, or unreadable figure text
was observed. The remaining log messages are benign underfull box warnings.

## Dataset integrity and split shift

The precomputed train/validation/test graph files are structurally clean: all
`5,002` graphs have finite targets and coordinates, valid nonempty edge indices,
and no duplicate exact targets. The test split is nevertheless compositionally
different from validation. Its median graph has 12 atoms and 326 directed edges,
versus 13 atoms and 416 edges in validation; its `2+` component standard
deviations differ by factors from `0.265` to `2.435`. This is a split shift, not
evidence of corrupted labels or a small set of invalid extremes.

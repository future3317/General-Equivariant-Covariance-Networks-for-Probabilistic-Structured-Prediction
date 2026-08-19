# E2 ITOP information-sufficiency and mechanism audit

Date: 2026-08-08  
Code commits: `526d79e` (probe), `d4af7d6` (perturbation diagnostic),
`382e083` (validation-selected component control)

## Phase boundary

E1 remains closed. This audit did not train E3, change the manuscript main
results, fit learned mixture gates, add component-specific scatters, or expose a
public finite-mixture compiler primitive. Top is cross-view evaluation only.

## Result matrix

| Evidence state | Question | Result | Evidence |
|---|---|---|---|
| Existing E1 evidence | Does ITOP K=2 improve the held-out density? | Yes on Side: exact NLL improves by about 2.31 for seeds 42/43/44; projection calibration improves, while Energy Score is essentially neutral. Top improves only slightly and remains severely rejected. | `docs/e1_frozen_distribution_results_20260808.md` |
| Existing E2 evidence | Does K=2 exhibit balanced target-dependent assignment? | No. Side median responsibility entropy is 0.126 after normalization, median maximum responsibility is 0.9996, and 98--99% of frames select one component with seed-dependent label switching. Median `||delta||/||residual||` is 0.043 and the posterior-assigned residual cosine is near zero. | `results/e2_itop_382e083/itop_mixture_mechanism/mechanism_audit.json` |
| New E2 evidence | Does raw inference-time geometry add stable Side-IID information beyond legal frozen-H invariants? | Not decisively. H+raw changes test MSE from 0.0024660 to 0.0024360, but the paired bootstrap 95% interval for `MSE(H+raw)-MSE(H)` is `[-7.25e-5, 1.26e-5]`, which crosses zero. Spearman changes from 0.318 to 0.345 and risk-coverage AUC from 0.0634 to 0.0611. Raw-only is weaker on Side. | `results/e2_itop_526d79e/itop_information_probe/probe_results.json` |
| New E2 evidence | Does the same probe transfer to Top? | No. All linear probes have strongly negative Top R2. Raw-only retains a modest Spearman signal (0.268), but H+raw is worse than raw-only and has Spearman -0.146. This is domain-shift behavior, not calibration evidence. | same probe artifact |
| New E2 evidence | Does input-perturbation pushforward variance track held-out frame error? | No. On Side, frame-level Spearman is 0.062 for missing block, 0.016 for point dropout, and 0.022 for quantization noise. On Top it is 0.024, 0.057, and -0.031. These covariances measure local input sensitivity, not predictive or epistemic covariance. | `results/e2_itop_d4af7d6/itop_observation_perturbation/diagnostics.json` |
| New E2 evidence | Is the equal-weight mixture itself needed for the Side NLL gain? | No. Selecting the globally dominant component using Side validation and evaluating that single shifted Student-t on Side test improves NLL over the frozen baseline by 2.92--2.97, versus 2.28--2.33 for the equal-weight mixture. The single shifted component is better than the mixture by about 0.645 NLL for every seed. | `results/e2_itop_382e083/itop_mixture_mechanism/mechanism_audit.json` |
| New E2 evidence | Does that validation-selected shift transfer to Top? | No. It worsens Top NLL by 0.73--0.80, whereas the symmetric mixture hedges the shift and improves by only 0.15--0.19. | same mechanism artifact |
| Inference | What generated the E1 Side mixture gain? | The evidence supports a frozen-mean structured-bias/density-mass surrogate, not demonstrated one-to-many conditional structure. The 1/2 mixture pays a log-weight penalty; a validation-selected single offset explains more of the IID gain. | Combined mechanism and component-control evidence |
| Inference | Is observation information loss the primary demonstrated cause? | No. The tested low-capacity raw descriptors do not add stable IID predictive MSE beyond H, and train-derived perturbation sensitivity does not rank frame errors. This rejects the tested form of the hypothesis, not every possible observation-aware representation. | Probe and perturbation evidence |
| Still unresolved | What remains after E2? | Model/function uncertainty and frozen-mean misspecification remain viable. Nonlinear raw-observation probes and learned observation-aware paths were not tested; the current linear probe cannot prove H is sufficient. | E2 scope boundary |

## Perturbation contract and artifact checks

- Intensities use Side-train observations only: median invalid-depth fraction
  `0.0409635` and sampled depth quantization step `0.001953125 m` (uniform-bin
  standard deviation `0.000563819 m`).
- Perturbations are structured missing-depth blocks, independent point dropout,
  and depth quantization noise. There are eight repeated deterministic forwards
  per sample and perturbation.
- Side test and Top each contain exactly 4,863 aligned sample IDs. All saved
  means and covariance matrices are finite; covariance matrices are symmetric.
- `pushforward_predictions.pt` SHA-256:
  `d9c9965ecb9a1d97ccfd7804ba3740f38c5a0b6e8c5d140ae2d8dbde299e18bc`.
- The pushforward object is a diagnostic sensitivity covariance. It was not
  added to learned scatter, temperature-scaled, or fitted to test residuals.

## E2 phase-gate decision

1. **Observation-information-loss gate: not passed.** H+raw has no significant
   Side test MSE increment under the predeclared linear probe, and perturbation
   trace does not track frame error.
2. **Finite-mixture topology gate: not passed.** Although H-only is broadly
   comparable to H+raw and K=2 improves proper Side NLL, a validation-selected
   single shifted component explains more of that gain. The result does not
   justify a public finite-mixture compiler primitive or claims of physical
   multimodality.
3. **Model/function-uncertainty gate: retained.** The tested H/raw/perturbation
   channels do not explain the severe single-model failure, while the useful
   IID density correction fails cross-view transfer. The next experimental
   phase should therefore be E3 true end-to-end multi-seed/bootstrap ensemble,
   evaluated with exact finite-mixture NLL. E3 was not started in this phase.


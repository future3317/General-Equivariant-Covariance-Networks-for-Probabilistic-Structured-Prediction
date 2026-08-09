# Uncertainty root-cause completion matrix

Date: 2026-08-09

This document evaluates the recommendations in the user-provided uncertainty
modelling brief (`不确定性建模意见.md`) by scientific evidence state. The brief is
an external working requirement and is intentionally not tracked by Git.
"Stopped by gate" is a completed decision, not an implementation defect. The
project should not run a rejected branch merely to turn every row into
"trained".

| Requirement | Implemented | Experimentally verified | Decision | Evidence |
|---|---|---|---|---|
| E0 elliptical-law falsification | Yes | Yes | Complete | `docs/uncertainty_root_cause_audit_20260808.md`; `results/uncertainty_root_cause_d945901` |
| Radial and random-projection PIT | Yes | Yes | Complete | `evaluation/metrics.py`; `evaluation/elliptical.py`; E0 artifacts |
| Whitened-direction and radius--direction diagnostics | Yes | Yes | Complete | `evaluation/elliptical.py`; E0 artifacts |
| E1 fixed-nu baseline reproduction | Yes | Yes | Complete | `docs/e1_frozen_distribution_results_20260808.md`; `results/e1_frozen_distribution_20260808` |
| Conditional nu from invariant features | Yes | Yes | Complete diagnostic; not promoted as final method | E1 dielectric and ITOP factorial |
| K=2 symmetric equivariant mixture | Yes, as controlled composition | Yes | Dielectric stopped at E1; ITOP interpretation closed at E2; public primitive frozen | E1 results; `docs/e2_information_sufficiency_results_20260808.md` |
| Exact finite-mixture NLL and sampling | Yes | Yes | Complete and reused; no second likelihood added | `evaluation/ensemble.py`; E1 and E3 artifacts |
| Mixture-aware projection PIT | Yes | Yes | Complete | `evaluation/elliptical.py`; E1 and E3 artifacts |
| Strict dielectric matched spectral control | Yes | Yes | Complete; SPD window not principal cause | E1 results and artifacts |
| E2 H/raw/H+raw sufficiency probes | Yes | Yes | Complete; tested low-capacity observation-aware route rejected | `docs/e2_information_sufficiency_results_20260808.md`; `results/e2_itop_*` |
| Observation perturbation pushforward covariance | Yes | Yes | Complete diagnostic; not added to predictive scatter | E2 results and artifacts |
| Existing ITOP K=2 mechanism audit | Yes | Yes, no training | Complete; evidence favors frozen-mean bias surrogate over two modes | E2 results |
| E3 independent model/function uncertainty | Yes, staged pilot orchestration and exact evaluator | Yes, two-seed development pilot | Strict gate failed; full three-seed confirmation stopped | `docs/e3a_fast_pilot_results_20260809.md`; `results/e3_pilot_c40898f` |
| E3 end-to-end joint training | Yes | Yes, seed-42 default LR and one low-LR control | Stopped: NLL--mean coupling | E3 pilot results |
| Bootstrap/subsample E3b | No | No | Correctly deferred because E3a gate failed | E3 protocol and result |
| Dielectric true deep ensemble | Evaluator exists | No | Correctly deferred because ITOP E3a gate failed | `scripts/evaluate_dielectric_ensemble.py` |
| Public finite-mixture compiler primitive | Design review only | No public-method experiment | Correctly frozen by E2 gate | `docs/finite_mixture_compiler_design_review_20260808.md` |
| Observation-aware predictive path | No public method | No | Correctly frozen by E2 gate | E2 result |
| Conditional flow / evidential head | No | No | Not triggered by phase gates | E1--E3 decisions |
| Repeated dielectric protocol labels | Not available | No | External evidence gap; cannot be manufactured | Dataset audit and E0/E1 reports |
| Typed uncertainty-source/identifiability semantics | No source-level schema; design review complete | No | Genuine public-API gap; implementation awaits review | `docs/uncertainty_source_semantics_design_20260809.md` |
| Objective-coupling control | Yes, by reusing the generic faithful objective | Yes, development pilot and full-data Stage A | Development gains did not pass the predeclared full-data gate; expansion stopped | `docs/objective_coupling_pilot_results_20260809.md`; `docs/objective_coupling_full_confirmation_results_20260809.md`; `results/objective_confirmation_7d42672` |

## What has actually been learned

1. Correct SPD algebra and full covariance coordinates do not establish a
   correct predictive law.
2. Dielectric has a real radial-tail mismatch, but conditional nu does not fix
   directional misspecification and cannot identify physical aleatoric
   covariance from one fixed-protocol label per structure.
3. ITOP's simple K=2 gain is not credible evidence of two physical pose modes;
   a validation-selected one-sided correction explains the IID gain better and
   does not transfer to Top.
4. The tested low-capacity raw-observation descriptors and perturbation
   covariance do not explain the remaining error.
5. Independent ITOP means help through model averaging, while their spread is
   not an error-aligned epistemic score.
6. Joint heteroscedastic NLL training can buy better density by degrading the
   mean. The faithful gradient boundary prevents that collapse and was positive
   in two development seeds, but its full-data Side run improved NLL by only
   0.571 while MPJPE and Energy Score worsened. It therefore failed promotion.

## Work that must remain stopped

Do not run more Full/Low-rank/Graph families, mixture gates, component-specific
scatters, OOF folds, temperature/conformal repairs, larger ensembles, or
additional learning-rate sweeps under the current hypotheses. These actions
would add cost without distinguishing a live root cause.

## Minimal remaining work

The formal full-data, 512-point faithful Stage A is complete and negative under
its predeclared multi-score gate. The matched ordinary arm was therefore not
launched, and no further GPU experiment is currently justified by the original
brief. The manuscript main results remain unchanged.

Separately, the typed uncertainty-source/identifiability schema is a real
compiler-level gap. It should first receive a minimal interface design showing
how a certificate records target, conditioning information, latent variables,
identifiability evidence, and calibration scope without claiming that the
compiler proves calibration. It does not require GPU training.
The design review is recorded in
`docs/uncertainty_source_semantics_design_20260809.md`; public implementation
remains gated on API review.

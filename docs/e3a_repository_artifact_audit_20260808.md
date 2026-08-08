# E3a repository and artifact audit

Date: 2026-08-08  
Scope: ITOP true end-to-end model/function uncertainty audit after E0--E2.

## Evidence matrix before E3a

| Requirement | Existing implementation | Credible experiment artifact | E3a action |
|---|---|---|---|
| Exact equal-weight finite Student-t mixture NLL | Yes: `evaluation/ensemble.py::finite_mixture_log_prob` / `ensemble_nll` | Unit tested; E1 frozen K=2 uses the same logsumexp semantics | Reuse unchanged |
| Finite-mixture sampling and Energy Score | Yes: `sample_ensemble`, `energy_score_from_samples` | Unit tested; E1 diagnostics use the same sampling semantics | Reuse unchanged |
| Mixture-aware projection PIT | Yes: `evaluation/elliptical.py::mixture_projection_pit` | Unit tested; E1 artifacts | Reuse unchanged |
| Within/between moment decomposition | Yes: `combine_ensemble_moments` | Unit tested; dielectric evaluator reports aggregate traces | Reuse the formula and label the between term as model/function spread |
| ITOP deterministic three-seed evaluator | Yes: `scripts/evaluate_itop_ensemble.py` | It evaluates deterministic predictions only and explicitly sets NLL to null | Do not reuse as a probabilistic-density evaluator |
| ITOP end-to-end independent-member trainer | No | No server E3 member runs or ensemble directories found | Add only `phase=end_to_end` plus a thin three-member runner |
| ITOP Full Student-t ensemble evaluator | No | No artifact | Add a task-neutral artifact combiner using the existing exact primitives |
| Resume/provenance for individual members | Yes in `scripts/train_itop.py` | Existing runs save source, dataset-cache hash, contracts, checkpoints and prediction hashes | Reuse; evaluator validates independent initialization and member provenance |
| Bootstrap/subsample members | No dedicated ITOP E3 protocol | No | Deliberately deferred to E3b |

## Artifact audit result

Server inspection found no complete end-to-end independent ITOP Full Student-t
member set. The available three-seed ITOP results are frozen-head/full-head
controls, and thus must not be relabeled as E3 evidence. Existing deterministic
three-seed logic is likewise not a continuous predictive density.

## Minimal implementation decision

The only code additions permitted for E3a are:

1. `end_to_end` in the existing ITOP training state machine. It initializes the
   Full Student-t model independently, trains all parameters, and rejects every
   input checkpoint or frozen feature cache.
2. A sequential single-GPU orchestration script for exactly three distinct
   seeds with identical Full Student-t fixed-nu controls.
3. A probabilistic ensemble evaluator that validates member provenance and uses
   the existing exact logsumexp, sampling, Energy Score, projection PIT and
   moment-decomposition utilities.

No new mixture likelihood, no moment-matched mixture NLL, no K=2 readout, no
conditional-nu ITOP model, no observation-aware branch and no public compiler
primitive is added.

## Predeclared E3a gate

Model/function uncertainty is supported only if the exact three-member density
beats all or most individual members on held-out Side NLL, improves Energy Score
or mixture-aware projection calibration, and produces non-degenerate
between-member spread that relates to error without being one-member dominated.
Top remains cross-view evaluation only. A failure of this gate stops the
deep-ensemble line rather than increasing ensemble size or introducing learned
weights.


# Reviewer claim audit

This record separates corrections that are logically forced by the current
implementation from experiments that would introduce a new scientific
variable. It is part of the provenance for the revision and does not treat an
unfinished run as evidence.

| Review point | Assessment | Action |
|---|---|---|
| Reference/active coordinates were conflated | Correct | Compiler audit now reports mean-target, total-target, and operator-coordinate dimensions separately. |
| `log-volume` named the centered spectral scalar | Correct | The scalar is documented as common mean log-eigenvalue/log-scale; `volume_min/max` remain compatibility schema names. |
| The whitening statistic was a pure angular defect | Correct | `whitened_second_moment_defect` is canonical; the old function remains a compatibility alias. |
| GoeCTP numbers were directly comparable | Correct | The manuscript now states that data construction, filtering, split, and metric protocols differ. |
| Berman et al. were described as proving too much | Correct | Related work now describes symmetry-aware calibration analysis without claiming calibration invariance or universal validity of post-hoc methods. |
| Raw-space moments were implied by log-space Student-t outputs | Correct | The contract now states that nonlinear pushforward moments may not exist for Student-t tails. |
| Dielectric scatter was physical aleatoric covariance | Correct concern | The existing conclusion is retained: it is surrogate predictive-distribution geometry under a single fixed-protocol label. |
| Graph Student-t isolated both graph and heavy-tail effects | Correct | Added executable `independent_student_t` and parameter-matched `low_rank_student_t` frozen-head arms. Results remain pending server execution. |
| Elasticity smoke MAE supported learning generality | Correct concern | The numerical MAEs were removed from the claim; the run is described only as an integration-path check. |
| Dense projector was a performance contribution | Correct concern | The manuscript now calls it an exact backend case study and states that algebraic exactness does not imply speed. |
| `AutoBudget` was statistical model selection | Correct | API and manuscript document it as an explicit compiler parameter-budget policy. |
| EIOC was a public compiler family or a demonstrated dielectric repair | Not supported by current code/evidence | It remains a prototype with a controlled synthetic intervention check; no real-data repair claim is made. |
| Learned `nu`, faithful joint modes, and more OOF folds were required | Not required for the current core claim | They would be new training protocols, so they are not mixed into the factorial experiment. |
| Finite-precision SPD behavior needs explicit qualification | Correct concern | Added `spd_contract` fields to compilation reports and ran `scripts.audit_spd_finite_precision`: 72 CPU cases across FP64/FP32/BF16 and logits -100/-1000/-10000. Zero-floor cases can become numerically singular; positive floors are reported as family parameters, not hidden jitter. |

The new ITOP factorial is deliberately frozen-head, single-seed, and uses the
same deterministic backbone, cache, split, optimizer, stopping rule, and
evaluation protocol for every arm. It can distinguish operator-family and
radial-law effects without reopening the rejected OOF/pseudo-W main line.

The finite-precision audit is stored at
`results/spd_finite_precision_audit.json`. Its 35/72 strict-SPD cases and
37/72 non-strict cases are an expected diagnostic of zero-floor or very small
Cholesky diagonals under finite arithmetic; they are not evidence that the
real-arithmetic compiler certificate is false.

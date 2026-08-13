# ITOP reviewer controls

## Protocol

The controls reuse the audited frozen ITOP representation and deterministic
mean.  Uncertainty heads are trained on Side only, selected by Side validation
NLL only, and evaluated on the complete Side and Top test sets.  Top is never
used for early stopping, hyperparameter choice, or model selection.  All runs
use fixed Student-t degrees of freedom `nu=5`, batch size 16,
AdamW learning rate `5e-4`, weight decay `1e-5`, at most 60 epochs, and patience
5.

The topology control compares the registered 15-node, 14-edge skeleton tree
with a degree-matched shuffled tree.  Both compiled Graph-t heads have 174
active operator coordinates.  The shuffled tree shares no edge with the true
skeleton.  The exact controlled comparison is seed 42, for which topology is
the only changed declaration.  The shuffled seeds 43/44 keep split seed 42 and
vary uncertainty-head initialization.  The pre-existing true-Graph seeds
43/44 use their respective Side split seeds, so their comparison is a
robustness check rather than an additional strictly paired topology contrast.
Every run shares the frozen representation, mean, complete Side/Top test IDs,
and validation-only rule.

The conventional control is a one-seed fixed-coordinate diagonal Student-t
head.  It bypasses the compiler and predicts 45 coordinate-wise log scatters
from the same pooled frozen features.

## Audited evidence

All saved predictions are finite.  FP64 scatter/precision materialization has
strictly positive minimum eigenvalues for every run and view.  Values below are
mean +/- sample standard deviation across seeds 42/43/44 unless marked seed 42.

| Model | Side NLL | Top NLL | Side Energy | Top Energy | Side MACE | Top MACE | Side Cov90/95 | Top Cov90/95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| True Graph-t | -55.936 +/- 0.010 | 4.067 +/- 1.895 | 0.726 +/- 0.000 | 2.485 +/- 0.039 | 0.128 +/- 0.006 | 0.678 +/- 0.019 | 0.804 / 0.911 | 0.131 / 0.183 |
| Shuffled Graph-t | -34.949 +/- 0.094 | 30.887 +/- 0.557 | 0.726 +/- 0.000 | 2.548 +/- 0.032 | 0.133 +/- 0.003 | 0.702 +/- 0.018 | 0.800 / 0.906 | 0.104 / 0.155 |
| No-edge Student-t (seed 42) | -27.050 | 41.130 | 0.728 | 2.442 | 0.152 | 0.645 | 0.776 / 0.890 | 0.168 / 0.226 |
| Fixed-coordinate diagonal-t (seed 42) | -27.764 | 72.809 | 0.727 | 2.368 | 0.140 | 0.769 | 0.796 / 0.900 | 0.022 / 0.036 |

In the exact seed-42 control, the true skeleton improves NLL over the
degree-matched shuffled tree by 20.901 on Side and 28.996 on Top.  The
three-seed aggregates differ by 20.987 and 26.820, respectively, with small
within-family seed variation.  The seed-42 result isolates declared skeleton
topology from coordinate budget and generic tree sparsity; the additional
seeds show that the conclusion is not a fragile initialization event, while
not being strictly paired for Side split.  The control does not establish
calibrated Top uncertainty: even true Graph-t remains severely under-covered.

The fixed-coordinate baseline offers no favorable statistical/equivariance
trade-off.  Its seed-42 Top NLL is 72.809 and Top Cov90/95 is 0.022/0.036.  On
512 saved Side features and eight deterministic random rotations, its scatter
rotation-consistency error has mean relative Frobenius norm 31.969 and maximum
964.298.  This is retained as a one-seed negative diagnostic, not a formal
multi-seed baseline ranking.

## Supported inference

In the matched seed-42 experiment, under a fixed representation, mean,
parameter budget, radial law, split, and selection rule, the true skeleton
topology materially improves predictive likelihood relative to a
degree-matched shuffled tree.  Initialization-robust aggregate results support
the same ordering.  The evidence rejects an explanation based only on compact
regularization.

## Rejected explanations and unresolved issues

- Rejected: an arbitrary 14-edge tree is sufficient to explain Graph-t's
  likelihood result.
- Rejected: an ordinary fixed-coordinate diagonal head provides a competitive
  practical alternative under the matched protocol.
- Unresolved: none of the tested heads repairs the severe Top-view calibration
  failure caused by Side-only supervision and observation shift.
- Not claimed: the graph family is universally preferable, or better NLL
  implies calibrated OOD uncertainty.

The machine-readable source is
`results/itop_reviewer_controls_2c7cb38/control_audit.json`; large checkpoints
and predictions remain in the preserved server result root and are identified
by `artifact_manifest.json`.

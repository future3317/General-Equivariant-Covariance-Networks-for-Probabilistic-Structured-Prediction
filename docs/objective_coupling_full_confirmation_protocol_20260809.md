# ITOP objective-coupling full-data confirmation protocol

Date: 2026-08-09

Status: predeclared staged confirmation. This protocol does not authorize
additional uncertainty families, ensemble members, or manuscript claims before
the gate is evaluated.

## Hypothesis

The positive 2,487-frame/256-point pilot reflects a real Side-IID training
mechanism: separating mean MSE gradients from detached-feature residual-NLL
gradients improves the proper predictive density without degrading the mean.

## Existing evidence and reusable control

The two-seed development pilot is recorded in
`objective_coupling_pilot_results_20260809.md`. The full-data control is the
existing seed-42 frozen Full Student-t checkpoint:

- 17,991 Side training frames, 512 points, fixed `nu=5`;
- Side test and Top test each contain 4,863 samples;
- checkpoint SHA-256:
  `76038afb3b720395cb9fbc5441e047fbe176709d6787475fa39f1e8cd1a89adb`;
- deterministic checkpoint SHA-256:
  `85e46daac36ea6fd04da518d4c86411e4add2723f62c57d933824433c137363a`;
- dataset-cache SHA-256:
  `94901c5488c6d7a30cddb1d87334b84a8640d2be90ca9ff121669bb6a4660269`;
- frozen Side MPJPE/NLL/Energy:
  `22.4167 cm / -70.8909 / 0.72368 m`;
- frozen Top MPJPE/NLL/Energy:
  `70.2234 cm / 9.6335 / 2.50011 m`.

The checkpoint was produced by clean source commit `f9e02f3`. Reuse is valid
only if the current model loads it strictly and its architecture, distribution,
operator schema, data root, point count, and cache metadata remain matched.

## Stage A: faithful arm only

Start from the frozen checkpoint and change only gradient routing:

- backbone and direct mean receive MSE gradients;
- covariance projection receives Student-t NLL gradients through detached
  features and detached residuals;
- compiled lifting receives no gradient in this stage;
- evaluation remains the ordinary proper Full Student-t density.

Keep seed and split seed 42, Full SPD operator, fixed `nu=5`, optimizer, learning
rate `5e-4`, weight decay `1e-5`, FP32 operator/NLL algebra, BF16 backbone
autocast, batch size 16, and validation-only checkpoint selection. Limit the
run to three epochs with patience two. Top is OOD evaluation only.

## Stage-A gate

All artifact, source, checkpoint, cache, finite-prediction, FP64 scale
materialization, and 4,863-sample checks must pass. On Side, require all three:

1. MPJPE does not worsen by more than `0.25 cm` from the frozen control;
2. proper NLL improves by at least `1.0` nat per frame;
3. Energy Score improves by at least one percent.

The thresholds are predeclared to avoid treating negligible numerical movement
as confirmation. Coverage alone cannot pass the gate. Top cannot rescue a Side
failure and cannot select a checkpoint.

If Stage A fails, stop objective-coupling expansion. If it passes, run one
matched ordinary-joint arm from the same frozen checkpoint for at most three
epochs. Do not add seeds before that within-checkpoint causal comparison is
complete.

## Interpretation boundary

A positive Stage A supports an IID optimization correction, not calibrated OOD
uncertainty. Opposite or weak Top behavior must be reported as cross-view
failure. No result identifies physical aleatoric covariance, revives mixture or
ensemble topology, or validates a public uncertainty-source schema.

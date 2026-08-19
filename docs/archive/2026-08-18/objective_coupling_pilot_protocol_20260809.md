# ITOP objective-coupling pilot protocol

Date: 2026-08-09

Status: predeclared development-only falsification after the E3 fast-pilot
gate. This is not E3b and cannot enter manuscript main results.

## Hypothesis

The current joint Student-t NLL improves density by sending a high-variance NLL
gradient through the mean/backbone path, degrading the mean while the scatter
absorbs the resulting residual. If this coupling is causal, preserving an MSE
mean objective and preventing covariance NLL gradients from entering the mean
feature path should retain the frozen mean while allowing useful scatter
adaptation.

## Existing evidence

- Seed-42 joint validation NLL improved monotonically from the frozen -52.8814
  to -85.3556 while MPJPE degraded from 30.2100 cm to 47.3761 cm and the mean
  gradient norm rose from 23.20 to 314.26.
- A ten-times-lower learning rate slowed but did not remove the same trend.
- `StructuredProbabilisticPredictor.forward_faithful_from_features` already
  implements the required MSE mean plus detached-feature residual NLL and is
  gradient-tested in the dielectric pipeline. No new likelihood or SPD map is
  required.

## Exact intervention

Use only seed 42 and the existing selected frozen Full Student-t checkpoint.
Run at most three joint epochs at the original learning rate 5e-4. The sole
change is the existing faithful gradient boundary:

- mean/backbone receive MSE gradients;
- covariance projection receives Student-t NLL gradients;
- NLL sees a detached current-mean residual and detached backbone/operator
  features;
- the fixed `nu=5`, Full SPD program, data, split, point count, optimizer,
  precision, and validation-only selection remain unchanged.

The ITOP trainer must record
`training_objective=mse_mean_plus_detached_feature_residual_nll`. Evaluation
uses the ordinary proper Student-t density, not the composite training loss.

## Decisive gate

Stop after one to three epochs. The route is supported only if, relative to the
same frozen checkpoint, Side validation/test MPJPE remains within 1 cm while
proper NLL improves and Energy Score does not worsen. Top remains OOD-only and
must not select a checkpoint. Failure closes this objective intervention; do
not sweep weights or learning rates.

## Seed-43 confirmation gate

Only if seed 42 passes the gate above, repeat the identical three-epoch
intervention from seed 43's own frozen Full-t checkpoint. This is a paired
development confirmation, not a return to the stopped full E3 ensemble. No
hyperparameter changes are permitted. The mechanism is considered stable
enough for method-level follow-up only if seed 43 also avoids mean collapse and
improves Side NLL plus Energy Score without degrading Side MPJPE by more than
1 cm. A failure stops the objective route; a success permits design of a formal
full-data comparison, but does not launch it automatically.

## Interpretation

A positive result supports gradient coupling as an actionable training cause,
not calibration or physical aleatoric covariance. A negative result means that
detaching the NLL from the mean path is insufficient and the remaining problem
should be described as mean/model/distribution misspecification rather than
rescued with more covariance families.

# ITOP reviewer-control evidence — 2026-08-17

This report records the read-only audit of the completed ITOP topology controls.
It is an evidence ledger, not a model-selection record.  Top-view data were
used only for final evaluation.

## Protocol and artifact status

- Formal control audit: `results/itop_reviewer_controls_2c7cb38/control_audit.json`.
- Split-matched replication: `results/itop_reviewer_controls_matched_20260816/`.
- Matched topology audit: `results/itop_topology_pairing_audit_20260817.json`.
- Frozen backbone/cache: seed-42 deterministic checkpoint and feature cache.
- Distribution: fixed Student-$t$, $\nu=5$.
- Selection: Side train/validation only; Top evaluation only.
- Required compact artifacts are present for seeds 42--44 where applicable:
  `args.json`, `environment.json`, `history.json`, `metrics.json`,
  `compilation.json`, `provenance.json`, and `train.log`.
- For the matched replication, both seed 43 and seed 44 prediction files are
  finite, and both Side/Top scatter audits report strictly positive minimum
  eigenvalues.

## Formal three-seed topology comparison

Mean $\pm$ sample standard deviation over seeds 42--44 from the completed
control audit.  `NLL` and `Energy` are lower-is-better; MACE and coverage are
reported descriptively and are not used as sole success criteria.

| Head | Side NLL | Top NLL | Side Energy | Top Energy | Side MACE | Top MACE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| True Graph-$t$ | $-55.936 \pm 0.010$ | $4.067 \pm 1.895$ | $0.726 \pm 0.000$ | $2.485 \pm 0.039$ | $0.128 \pm 0.006$ | $0.678 \pm 0.019$ |
| Shuffled Graph-$t$ | $-34.992 \pm 0.060$ | $32.107 \pm 0.865$ | $0.726 \pm 0.000$ | $2.547 \pm 0.030$ | $0.135 \pm 0.004$ | $0.700 \pm 0.019$ |

The true-versus-shuffled result therefore supports a topology-specific
likelihood effect under the audited protocol.  Pooled paired per-frame NLL
differences (shuffled minus true) are $+20.987$ with 95\% bootstrap CI
$[20.891,21.079]$ on Side and $+26.820$ with CI $[26.741,26.899]$ on Top
(14,589 paired frames per view).  It does not support a claim of cross-view
calibration: Top coverage remains strongly below nominal for both structured
heads.

## Secondary one-seed controls

| Head | Side NLL | Top NLL | Side Cov90 / Cov95 | Top Cov90 / Cov95 | Rotation audit |
| --- | ---: | ---: | ---: | ---: | --- |
| No-edge Independent-$t$ | $-27.050$ | $41.130$ | $0.776 / 0.890$ | $0.168 / 0.226$ | n/a |
| Fixed-coordinate diagonal Student-$t$ | $-27.764$ | $72.809$ | $0.796 / 0.900$ | $0.022 / 0.036$ | finite, but mean relative error $31.97$, max $964.30$ |

The fixed-coordinate row remains a negative rotation-consistency diagnostic,
not a competitive equivariant baseline.

## Pairing and artifact conclusion

The seed-42 control and the seed-43/44 replication artifacts were audited
together with the corresponding True Graph-$t$ runs.  For every seed, the
protocol fields, frozen-cache fields, effective split seed, frame order, and
targets match exactly; each view contains 4,863 paired frames.  All saved
predictions are finite and the saved scatter materializations are strictly SPD.
The resulting contrast changes only the declared topology within each matched
seed/split.  Minimum-eigenvalue details remain in the raw artifacts rather than
being used as a headline performance claim.

## Decision

The formal three-seed true-versus-shuffled audit is clean and paper-relevant;
the paired contrast replaces the former split-42-only wording.  No new method,
hyperparameter, or calibration claim is introduced from this audit, and
Graph-$t$ Structured diagnostics, including panel (a), remain unchanged.

# ITOP objective-coupling full-data confirmation results

Date: 2026-08-09

Status: **Stage-A gate failed; objective-coupling expansion stopped.** The
matched ordinary-joint arm was not launched.

## Existing evidence

On the 2,487-frame/256-point development protocol, faithful gradient routing
improved Side MPJPE, proper NLL, and Energy Score for seeds 42 and 43 relative
to their own frozen Full Student-t checkpoints. The predeclared full-data gate
therefore tested whether that result survived the formal 17,991-frame,
512-point seed-42 protocol.

## New full-data evidence

The faithful arm started from the existing frozen Full Student-t checkpoint
and changed only the training gradient boundary. The validation-NLL-selected
checkpoint was epoch 3.

| Split | Stage | MPJPE (cm) | NLL | Energy Score (m) |
|---|---|---:|---:|---:|
| Side IID | Frozen control | **22.4167** | -70.8909 | **0.72368** |
| Side IID | Faithful joint | 22.6936 | **-71.4623** | 0.73200 |
| Top OOD | Frozen control | 70.2234 | 9.6335 | 2.50011 |
| Top OOD | Faithful joint | **69.7946** | **8.1535** | **2.43732** |

Side changes relative to frozen were:

- MPJPE: `+0.2769 cm` (worse);
- NLL: `-0.5714` nat per frame (better, but below the required `1.0`);
- Energy Score: `+0.00831 m`, or approximately `+1.15%` (worse).

The predeclared gate required MPJPE degradation no larger than `0.25 cm`, NLL
improvement of at least `1.0`, and Energy improvement of at least one percent.
All three conditions were required. Stage A therefore fails. The Top movement
is useful cross-view diagnostic evidence but cannot rescue an IID gate or
select a checkpoint.

## Artifact audit

- clean source commit:
  `7d4267227fa4f1e71ee17685db4ea9fb73d4a87b`;
- input checkpoint SHA-256:
  `76038afb3b720395cb9fbc5441e047fbe176709d6787475fa39f1e8cd1a89adb`;
- dataset-cache SHA-256:
  `94901c5488c6d7a30cddb1d87334b84a8640d2be90ca9ff121669bb6a4660269`;
- Side/Top prediction counts: `4,863 / 4,863`;
- all prediction tensors finite;
- all six provenance artifact hashes match after server and local transfer;
- Side and Top FP64 scale materialization passed strict SPD checks;
- the training contract records validation-NLL selection and the explicit
  faithful `gradient_routing` boundary.

Evidence paths:

- server:
  `/home/workspace/lrh/RESULTS/Tpami/ITOP/objective_confirmation_7d42672`;
- local: `results/objective_confirmation_7d42672`.

## Supported inference

Faithful gradient isolation is a valid optimization control and prevented the
catastrophic mean collapse seen in ordinary development joint training. Its
benefit was not large or consistent enough on the formal Side protocol to
support promotion as the uncertainty solution or a manuscript main result.

## Rejected explanation and stopped work

The development gains do not establish a scale-robust objective correction.
Per the staged protocol, the ordinary full-data arm, additional seeds, learning
rate sweeps, mixture, ensemble, observation-aware, and E3b work remain stopped.
No manuscript main table is changed by this negative confirmation.

## Unresolved

The experiments still support multiple distinct failure axes--radial law,
directional law, observation shift, model averaging, and optimization--but no
tested uncertainty extension has passed its formal promotion gate. The only
remaining non-training item from the original brief is review of the proposed
typed uncertainty-source/identifiability API; its design document is not an
implemented public contract.

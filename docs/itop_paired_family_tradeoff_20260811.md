# ITOP paired Full-t versus Graph-t trade-off

Date: 2026-08-11

Status: **artifact-only audit complete.** No model was trained and Top was not used for selection.

## Contract

Seeds 42, 43, and 44 are frozen uncertainty-head runs sharing the same deterministic backbone and feature cache. Within each seed, Full-t and Graph-t use identical test targets, frame ordering, frozen means, fixed `nu=5`, and validation-only head selection. They are head/split seeds, not independent backbone initializations.

Per-sample proper NLL was recomputed through the existing `StudentTNLL.log_prob` and compiled SPD/precision maps. No moment-matched or second likelihood implementation was introduced. The point estimates first average the three seed-wise differences for each frame. The uncertainty audit additionally resamples the four subject clusters recovered from official compact label IDs. The official Zenodo schema defines each identifier as `XX_YYYYY` (person ID and frame number) and provides no action label or sequence-boundary field, so sequence boundaries are not inferred from frame-number gaps.

## Paired proper-score result

Graph-minus-Full mean NLL and paired 95% intervals are:

| View | Seed 42 | Seed 43 | Seed 44 | Across-seed frame mean |
|---|---:|---:|---:|---:|
| Side IID | +14.960 [12.822, 16.847] | +14.806 [12.856, 16.526] | +14.779 [12.470, 16.746] | +14.848 [12.743, 16.707] |
| Top OOD | -7.152 [-10.113, -3.761] | -2.139 [-2.652, -1.526] | -3.825 [-5.443, -2.467] | **-4.372 [-5.929, -3.021]** |

Full-t is decisively better on Side IID, while Graph-t is decisively better under the predeclared Top cross-view shift in every head seed. The result therefore supports a structured family trade-off, not universal Graph superiority and not Top calibration.

## Coordinates and measured resource trade-off

The existing matched RTX 4090 runtime audit used batch 16, 25 warmups, 100 repeats, eager FP32, identical cached Side features, and synchronized timing.

| Family | Active coordinates | Top NLL (seed mean) | Forward median | NLL-eval median | Fwd+bwd peak allocated |
|---|---:|---:|---:|---:|---:|
| Full-t | 1,035 | 8.439 | 1.485 ms | 7.911 ms | 40.77 MB |
| Graph-t | 174 | **4.067** | **1.210 ms** | 13.262 ms | **18.95 MB** |

Graph-t uses 83.2% fewer active coordinates and 53.5% less measured forward-backward peak allocation. It is on the Top-NLL/coordinate and Top-NLL/memory Pareto frontier. Its compiled operator forward is 18.5% faster in this microbenchmark, but exact graph-precision NLL evaluation is slower than Full (13.26 versus 7.91 ms); no blanket latency advantage is claimed.

## Supported inference and boundary

This strengthens the narrow manuscript statement that Graph-t provides a much smaller structured uncertainty family and better cross-view predictive likelihood under the audited frozen representation. It does not establish independent-model robustness, reliable OOD ranking, calibrated Top uncertainty, or a general speed advantage. The complete JSON evidence is `results/itop_paired_family_tradeoff_feb75b9.json`.

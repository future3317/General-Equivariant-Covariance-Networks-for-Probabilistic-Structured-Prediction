# Rank-4 Elasticity End-to-End Results (2026-08-11)

## Decision

The formal rank-4 elasticity gate **passes**. All nine preregistered arm/seed
cells completed on the official split with complete artifacts, finite held-out
predictions, validation-only checkpoint selection, and a clean common source.
Both compiled Student-t arms reconstructed strictly SPD scatter matrices in
FP64. This closes the former compile-only evidence gap: a 21-dimensional
rank-4 output, including the Full family's canonical \(\ell=8\) operator
coordinates, was compiled, trained, and evaluated end to end.

The result does not support a calibration claim. Full-t gives the better
proper likelihood, while rank-2 Low-rank-t gives better point error, Energy
Score, and coverage. Both families strongly fail the elliptical diagnostics.
The legacy Voigt protocol is not representation-compatible; it is therefore
kept as a historical stress test rather than pooled with the separate
representation-compatible chart audit below.

## Protocol and artifact audit

- Source: clean commit `feb75b9dbe01c7fe3defa7bd357bb1ef9cec71d0`.
- Data: official fixed files with identical hashes in all cells; 8,801 train,
  1,625 validation, and 799 test structures.
- Arms: deterministic mean, rank-2 Low-rank Student-t, and Full Student-t;
  Student-t arms use fixed \(\nu=5\).
- Seeds: 42, 43, and 44; batch size 16; at most 12 epochs; patience 3.
- Selection: validation criterion only. The recorded selected epoch equals the
  argmin validation epoch in every cell.
- Algebra: FP32 training and likelihood evaluation; independent FP64 scatter
  materialization for the numerical SPD audit.
- All nine cells contain `args.json`, `environment.json`, `schema.json`,
  `history.json`, `metrics.json`, `predictions.pt`, `best_model.pt`, and
  `train.log`; `study_manifest.json` records the complete 3-by-3 study.
- The 799 unique test IDs and their order are identical across all cells. All
  saved means, targets, and operator parameters are finite.
- The local mirror at `results/elasticity_end_to_end_feb75b9` contains all
  formal artifacts. SHA-256 checks agree byte-for-byte with the server copy.

| Arm | Output/operator audit | Active operator coordinates | Selected epochs | FP64 SPD |
|---|---|---:|---|---|
| Deterministic | 21-dimensional rank-4 mean | -- | 12 / 11 / 12 | N/A |
| Low-rank-t (rank 2) | 21-dimensional output; canonical program reaches \(\ell=8\) | 43 | 12 / 11 / 12 | 3/3 pass |
| Full-t | 21-dimensional output; canonical \(\ell=8\) operator representation | 231 | 9 / 11 / 4 | 3/3 pass |

Seed 44 Full-t stopped after epoch 7 because its validation optimum was epoch
4 and patience 3 was exhausted. This is the preregistered stopping rule, not a
failed cell.

## New evidence

Values are held-out test mean \(\pm\) sample standard deviation over three
initialization seeds. NLL includes the exact normalized Student-t constant.

| Arm | MAE (GPa) ↓ | NLL ↓ | Energy ↓ | Cov90 ↑ | Cov95 ↑ | Coordinates |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic | 16.610 ± 0.168 | -- | -- | -- | -- | -- |
| Low-rank-t | **15.241 ± 0.098** | 31.969 ± 0.043 | **6.999 ± 0.011** | **0.509 ± 0.008** | **0.574 ± 0.007** | **43** |
| Full-t | 17.530 ± 0.196 | **28.628 ± 0.490** | 7.187 ± 0.043 | 0.374 ± 0.057 | 0.456 ± 0.053 | 231 |

The rank-2 family uses 81.4% fewer operator coordinates than Full. This is a
representation-size result, not a speed claim: concurrent scheduling and
different early-stopping epochs make wall-clock comparisons unsuitable for a
kernel-level conclusion. Measured throughput was similar (19.92 ± 1.73 versus
19.38 ± 1.52 examples/s for Low-rank-t and Full-t).

### Elliptical-law diagnostics

| Arm | Radial KS ↓ | Direction defect ↓ | Projection max KS ↓ | Radius-direction permutation p |
|---|---:|---:|---:|---:|
| Low-rank-t | 0.394 ± 0.006 | 7.447 ± 0.225 | 0.406 ± 0.046 | 0.005 / 0.005 / 0.005 |
| Full-t | 0.531 ± 0.059 | **5.491 ± 0.021** | **0.365 ± 0.004** | 0.005 / 0.005 / 0.005 |

Full-t improves directional and projection diagnostics relative to rank-2
Low-rank-t, but its radial fit and coverage are worse. Every probabilistic
cell rejects radius-direction independence at the minimum 199-permutation
resolution. Algebraic validity and end-to-end trainability therefore do not
imply a statistically adequate elliptical predictive law.

## Representation-compatible full-image chart audit (2026-08-19)

The full rank-4 target was rerun with the registered full-image spectral chart

\[
f(\lambda)=\exp(\operatorname{asinh}\lambda),
\]

under representation-compatible target normalization. This is a separate
three-seed evidence contract, not a repair of or numerical ranking against the
legacy Voigt protocol. All seeds used the same 231-coordinate Full target,
three exact lifting edges through $\ell=8$, fixed $\nu=5$, and validation-only
selection. The compact audit is
`results/elasticity_stability_20260819/asinh_exp_formal_20260819/asinh_elasticity_audit.json`.

| Chart | MAE (GPa) $\downarrow$ | NLL $\downarrow$ | Energy $\downarrow$ | Cov90 $\uparrow$ | Cov95 $\uparrow$ | FP64 strict SPD |
|---|---:|---:|---:|---:|---:|---|
| $\exp(\operatorname{asinh}\lambda)$ | 13.333 $\pm$ 0.713 | 18.809 $\pm$ 0.389 | 2.788 $\pm$ 0.023 | 0.534 $\pm$ 0.049 | 0.657 $\pm$ 0.057 | 3/3 |

All predictions are finite, all three active and canonical reachability audits
record depth 3 with the $\ell=8$ obligation, and the independent FP64 minimum
eigenvalues are positive. These values establish representation-compatible
execution evidence for the full-image chart; they do not establish calibration
or a cross-normalization family ranking.

## Existing evidence

- The compiler already constructed the rank-4 elasticity output
  `2x0e+2x2e+1x4e` and the Full symmetric-operator representation reaching
  \(\ell=8\), but this was previously compile-time evidence only.
- The seed-42 pilot established finite training, schema validity, FP64 SPD
  reconstruction, and an acceptable batch-16 resource envelope before the
  formal run.
- The independent NumPy/SciPy teacher experiment had already removed circular
  teacher/learner validation for controlled recovery; it is logically
  separate from this real-data trainability test.

## Supported inference

1. The typed compiler supports genuine end-to-end training and proper-score
   evaluation for a high-order, 21-dimensional structured output; its practical
   evidence is not limited to rank-2 dielectric tensors or repeated-vector pose
   outputs.
2. Operator-family choice remains statistically consequential under the same
   output type, dataset, optimizer budget, and selection protocol. Full-t wins
   exact NLL, while rank-2 Low-rank-t wins Energy Score and point prediction.
3. Compactness is real at the operator-coordinate level: rank-2 Low-rank-t uses
   43 rather than 231 coordinates. No general latency or memory advantage is
   inferred from this concurrent experiment.
4. The compiler's algebraic guarantees remain distinct from statistical
   adequacy: all six Student-t cells are valid and numerically SPD, yet all six
   reject the fitted elliptical law.

## Rejected explanation

- The former statement that elasticity was only a compile-time construction is
  no longer accurate.
- Full expressivity alone does not dominate every predictive criterion: Full-t
  improves NLL but not MAE, Energy Score, or marginal coverage.
- Low-rank restriction alone does not repair calibration or conditional-law
  misspecification.

## Unresolved

- The experiment is not a point-prediction benchmark and does not establish
  materials-property SOTA.
- It does not identify the physical source of the remaining radial and
  directional misspecification.
- Runtime evidence is workload-level only. A claim about asymptotic or
  kernel-level speed would require isolated matched profiling and is not needed
  for the present scientific gate.
- No additional elasticity family sweep, temperature correction, or backbone
  expansion is needed for the current paper; the asinh chart is reported as a
  separate representation-compatible audit rather than merged with legacy
  Voigt values.

## Artifact hashes

The following exact SHA-256 values were recomputed after download and matched
against the server files.

| Cell | `best_model.pt` | `predictions.pt` |
|---|---|---|
| seed42 deterministic | `a66d0a531941edb19174e57968b58bf42d48e58b5adf1dc1d9f8ad9a7232cb04` | `62dfcbb6fd348286020f7fa6b4b96120c85e3f1a5cd611511faff4c7db10acc7` |
| seed42 Low-rank-t | `beb5b79442a6a6c22252431bf4f5baa7d089e7c18c227b2443c32c35d8e5b56b` | `21fa231d9869cb46a0d945d77a9d6f9015a61d2bb2b1ea93dfe28b93f64ddee5` |
| seed42 Full-t | `5d37e4d8b3048a414fd86d64e143870c642c2e4cd8dcd2fbfc37602739370821` | `d330874c595da4e05c79bba7b32dbc2004a0afa547376a1001898ee90066044d` |
| seed43 deterministic | `9eed9f043e9af3fb467d814adee8400972ade4bdcbe15b9378f510043e61b7fa` | `3f66fd68bef016c28b4f1178e1086074561f7086cee607f44f6d6ca8eda142f2` |
| seed43 Low-rank-t | `64c23bd11d31463af9de3aec84f4e103be7cec518b039a637e367ee95313f5e2` | `02cc560fafbc9057e7fcd193a520e494555a33680b1f793a1143524b939718cc` |
| seed43 Full-t | `8abeae22a2b067685b7099650bb52c502ef62d19c2b69a8a62b3a8ff5f0d048e` | `9aee4f444b0796216da57ba91d00e524d4c839d90d4854df861ef09e52b02627` |
| seed44 deterministic | `c29b62ff6628dd4a58c341d1c90224e9c2c73913151cd81a39eb37a520f03481` | `1e2322b72f29e881e5a9bffcf5e0bdbeb45c3bbf0422fee90a4e671af0c4e274` |
| seed44 Low-rank-t | `d8c79a0b15ff6b42d25eaafd8874af2ee5977ebc2fa8216d5faa852bc67adc58` | `cb3e5b9beb79d4c6756b110384956a4316ad09f805a527338746b56916815d9b` |
| seed44 Full-t | `a1820c48d731bb2fd98599c3d67a96f4824870fe0a1664e9c517f76fb008a553` | `d19d47b033096648ebf6b2112cb38b793fabeadec77319d20d0dca8d9bf9f4b1` |

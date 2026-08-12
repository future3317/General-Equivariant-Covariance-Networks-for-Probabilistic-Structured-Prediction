# Dielectric uncertainty-representation repair: A/B result

Date: 2026-08-12
Status: development-only, one seed, stopped after the predeclared repair gate

## Question

Does the remaining dielectric misspecification come primarily from insufficient
frozen representation information, or from the predictive law/operator shape?

The comparison keeps the dielectric split, targets, sample IDs, frozen mean,
conditional-\(\nu\) law, 21-coordinate centered spectral-window operator,
validation-only selection, exact Student-t NLL, and compiled SPD/exact-lowering
semantics fixed.

- **A:** frozen \(H\) + conditional \(\nu(x)\), the confirmed radial-repair baseline.
- **B:** A initialized with a zero-initialized uncertainty-only equivariant
  residual branch, with the mean detached from uncertainty loss.

B was initialized from A's checkpoint; it was not a new mean model. Both runs
used seed 42 and the same train/validation/test sample IDs.

## Held-out test comparison

| metric | A: frozen \(H\) + conditional \(\nu\) | B: uncertainty branch + conditional \(\nu\) | B − A |
|---|---:|---:|---:|
| normalized NLL | -2.8480 | **-2.9966** | -0.1486 |
| Energy Score | **0.4419** | 0.4462 | +0.0043 |
| Coverage50 | **0.4626** | 0.3523 | -0.1103 |
| Coverage90 | **0.7758** | 0.7544 | -0.0214 |
| Coverage95 | 0.8577 | **0.8612** | +0.0036 |
| MACE | **0.0762** | 0.1190 | +0.0427 |
| whitened second-moment defect | 5.6665 | **3.3544** | -2.3121 |
| radial PIT KS | **0.1488** | 0.2162 | +0.0673 |
| max \(|\rho_S|\) radius--direction | 0.4669 | 0.4559 | -0.0110 |
| radius--direction permutation p | 0.005 | 0.005 | unchanged rejection |
| scalar alignment proxy \(\mathrm{corr}(q,\|r\|^2)\) | 0.2635 | 0.2569 | -0.0066 |

The radius--direction test remains rejected for B. The small reduction in its
Spearman maximum is not a substantive diagnostic repair, and the scalar
uncertainty/error alignment proxy does not improve.

## Integrity checks

- train/validation/test sample IDs: identical between A and B;
- targets: identical;
- frozen means: identical to machine precision (maximum absolute difference 0);
- parameter shape: 21 coordinates in both runs;
- scale shape: \((N,6,6)\) in both runs;
- all saved mean, target, \(\nu\), parameter, and scale values: finite;
- B minimum FP64 scale eigenvalue: positive (`1.818e-4`); spectral condition
  remains finite and within the existing window control;
- B protocol records zero-initialized residual provenance from A;
- operator program hash, typed parameter representation, SPD certificate, and
  exact active-family lowering are unchanged;
- model selection used validation NLL only; no test/OOD quantity was used.

## Decision

**Stop representation repair.** B fails the promotion gate because the NLL and
whitened-defect gains are accompanied by worse Energy, Coverage50/90, MACE, and
radial PIT, with no alignment improvement and continued directional rejection.
No full-data run or three-seed confirmation is justified.

The supported conclusion is:

> Conditional \(\nu(x)\) repairs part of the radial/tail misspecification, but a
> one-seed uncertainty-only representation branch does not provide evidence
> that frozen representation information is the dominant remaining bottleneck.
> The remaining error is consistent with non-elliptical directional
> misspecification and/or a mismatch in the structured scatter law; this result
> does not distinguish those two causes.

Do not extend this route with more mixture components, flow/diffusion, wider
spectral sweeps, ITOP conditional-\(\nu\)/descriptor experiments, or additional
hyperparameter searches in this round. Keep A, B1, and B2 artifacts unchanged
for provenance.

Machine-readable comparison: `RESULTS/Tpami/stat_misspec_representation_20260812/ab_comparison.json`.
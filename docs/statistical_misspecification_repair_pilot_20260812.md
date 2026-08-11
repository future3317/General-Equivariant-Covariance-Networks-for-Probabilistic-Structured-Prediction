# Statistical misspecification repair: first pilot

This is a development-only, one-seed falsification study. It reuses the
existing frozen dielectric features, mean, split IDs, and centered spectral
window. The compiler planner, typed family, SPD map semantics, and exact
lowering are unchanged. The K=2 NLL is the exact normalized
`logsumexp(log pi_k + log p_k)` mixture; no moment-matched NLL is reported.

## Dielectric test split

| arm | NLL | Energy | coverage90 | coverage95 | MACE | whitened defect |
|---|---:|---:|---:|---:|---:|---:|
| fixed nu=5 | -2.625 | 0.4410 | 0.712 | 0.769 | 0.110 | 10.462 |
| global nu | -2.662 | 0.4430 | 0.722 | 0.776 | 0.106 | 9.960 |
| conditional nu(x) | **-2.848** | **0.4419** | **0.776** | **0.858** | **0.076** | **5.666** |
| K=2 shared-mean mixture | **-2.914** | 0.4423 | 0.826* | 0.884* | 0.131* | — |

The starred mixture coverage/MACE values are exact component-mixture
coordinate-wise marginal intervals, not joint elliptical coverage; they are
therefore not directly comparable to the single-elliptical rows. The mixture
projection PIT remains poor (`max KS=0.209`, 48 Bonferroni rejections,
`moment_matched=false`). For all single-law rows, the radius--direction
independence test still rejects (`p=0.005`).

## Answers to the pilot questions

1. **Is fixed nu the main cause?** Partly. Conditional nu gives the clearest
   improvement in proper NLL, marginal calibration, and whitened second-moment
   defect. The independence rejection remains, so fixed nu is not the whole
   misspecification.
2. **Does K=2 clearly repair the single elliptical law?** No. It improves the
   exact mixture NLL, but Energy does not improve and the mixture-aware
   projection PIT still fails. Stop before adding multimodal means or larger
   mixtures.
3. **Is the spectral window the bottleneck?** Not tested in this pilot. Because
   the conditional radial law already improves the main diagnostics without
   changing the window, widening the window is deprioritized; a separate
   one-seed window control is only justified if the repair decision requires it.
4. **Do observation descriptors improve ITOP Top calibration?** Unresolved.
   The server workspace exposes the frozen Full `H, mu` cache but no raw
   geometry/depth cache or Graph frozen cache. The attempted Full ITOP exact
   map reconstruction stayed in CPU planner initialization without producing
   artifacts, so those temporary processes were stopped and no ITOP claim is
   made.

## Decision

Keep `conditional_nu` as the only promising repair candidate for a later
three-seed/full-data confirmation. Do not promote the pilot numbers to the
paper and do not expand the mixture direction until a proper held-out
mixture-aware calibration improvement is demonstrated. The machine-readable
artifact report is
`results/stat_misspec_pilot_868baac/comparison.json`.

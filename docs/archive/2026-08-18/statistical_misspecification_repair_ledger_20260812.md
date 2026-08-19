# Statistical misspecification repair ledger

All new rows use the existing frozen dielectric split and source cache. The
ITOP rows below are existing evidence only; no Top/OOD value was used for
selection.

| baseline → modification | hypothesis | subset result | diagnosis | next action |
|---|---|---|---|---|
| fixed Student-t, \(\nu=5\) → global \(\nu\) | fixed radial tail is too rigid | test NLL -2.625 → -2.662; MACE 0.110 → 0.106; defect 10.462 → 9.960; Energy 0.4410 → 0.4430 | weak radial improvement; Energy slightly worse | do not promote alone |
| fixed Student-t → conditional \(\nu(x)\) | tail mismatch varies with frozen invariant features | test NLL -2.625 → -2.848; Energy 0.4410 → 0.4419 in seed 42; Coverage90 0.712 → 0.776; Coverage95 0.769 → 0.858; MACE 0.110 → 0.076; defect 10.462 → 5.666 | supported; stable across seeds; radius--direction independence still rejected; no separate alignment endpoint is promoted | full-data confirmation completed for seeds 42/43/44 |
| fixed Student-t → K=2 shared-mean mixture | one elliptical component is too restrictive | exact mixture NLL -2.914, but Energy 0.4423 and mixture PIT (`48` Bonferroni rejections) do not improve | rejected as mixture evidence: weights, \(\nu\), scatter, and means are exactly collapsed across components | stop mixture expansion |
| centered spectral window → wider/reference window | spectral bounds are the bottleneck | not run after radial pilot | not yet supported; conditional law improves without changing window | deprioritize window sweep |
| frozen ITOP Full-t → observation descriptors / Graph conditional law | frozen uncertainty representation lacks observation-quality information | not executable from current workspace: no raw geometry/depth or Graph frozen cache with selectable Side validation artifacts | unresolved, not a negative result; existing Full-t is Side NLL -70.891 / Top NLL 9.633, Side Cov90 0.832 / Top Cov90 0.159, Top MACE 0.500 | acquire/prepare the declared label-free geometry cache before any ITOP repair run |

## Dielectric three-seed confirmation

The conditional-\(\nu\) confirmation used validation-only NLL selection and
seeds 42/43/44. The reported values are test means ± seed standard deviation;
the fixed baseline is the same frozen fixed-\(\nu=5\) artifact.

| method | NLL | Energy | Coverage90 | Coverage95 | MACE | whitened defect | radial PIT KS | independence p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed \(\nu=5\) | -2.625 | 0.4410 | 0.712 | 0.769 | 0.110 | 10.462 | 0.214 | 0.005 |
| conditional \(\nu(x)\) | **-2.848 ± 0.0002** | **0.4396 ± 0.0020** | **0.776** | **0.858** | **0.076 ± 0.0002** | **5.661 ± 0.006** | **0.149 ± 0.0001** | 0.005 |

All three confirmation artifacts had finite predictions, FP64 SPD minimum
eigenvalue `2.85e-4`, exact active-family lowering, and
`ood_used_for_selection=false`. The learned degrees of freedom remained
interior (`nu_min≈2.28`, `nu_median≈3.07`, `nu_max≈3.85`), so the result is
not an optimization hit on the parameter bounds. The uncertainty/error
alignment Spearman statistic was `-0.013` for both fixed and conditional
single-law predictions, so radial flexibility did not add observation-level
ranking information.

## Decision

The only modification that survives the causal and stability checks is
conditional radial flexibility. It is worth retaining as a separate audited
dielectric result, but it does **not** repair the directional misspecification:
the radius--direction test remains rejected; no separate uncertainty/error
alignment endpoint is promoted because the scatter representation remains frozen.

No further dielectric exploration is justified in this round. ITOP repair is
not promoted or rejected until the missing raw observation/validation
artifacts are available. The machine-readable pilot and confirmation reports
are `results/stat_misspec_pilot_868baac/comparison.json` and
`results/stat_misspec_confirm_868baac/confirmation_comparison.json`.

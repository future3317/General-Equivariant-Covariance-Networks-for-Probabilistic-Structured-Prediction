# ITOP full-study decision and root-cause diagnosis

Scope: seed 42, all 17,991 valid side-train frames, 512 sampled points, one
RTX 4090, side-test as IID and top-test as cross-view OOD. Test views contain
the same 4,863 valid frame IDs. The report is based on completed prediction
artifacts at the final single-seed study output; the graph Student-t joint
ablation was intentionally stopped after epoch 6 and is not treated as a
final result.

## Decision

Do not continue the same side-only joint fine-tuning or launch more repeated
seeds for this question. The completed frozen-head comparison is sufficient to
select graph Student-t as the main ITOP uncertainty head:

| Model | Side MPJPE (cm) | Side NLL | Top MPJPE (cm) | Top NLL | Side/top uncertainty AUROC |
|---|---:|---:|---:|---:|---:|
| Deterministic | 22.406 | -- | 70.277 | -- | -- |
| Independent Gaussian, frozen | 22.406 | -19.969 | 70.277 | 772.924 | 0.188 |
| Graph Gaussian, frozen | 22.406 | -50.600 | 70.277 | 819.485 | 0.001 |
| Graph Student-t, frozen | 22.406 | -55.955 | 70.277 | 2.330 | 0.803 |
| Independent Gaussian, joint | 21.979 | -20.198 | 67.407 | 4,988.677 | ~0 |

The small joint-independent MPJPE improvement is not worth the severe loss of
proper OOD likelihood and uncertainty ranking. Frozen graph Student-t is the
only completed head that combines a strong side proper score with a useful
cross-view uncertainty signal. This is an OOD diagnostic, not a calibration or
SOTA claim.

## Why the training was repetitive

The runner deliberately trained several uncertainty families on the same
deterministic backbone, then automatically selected a graph family and
resumed joint fine-tuning whenever an incomplete `last_state.pt` existed.
That is useful for an initial ablation, but once the frozen comparison had
separated the candidates, continuing the same joint stage no longer answered a
new scientific question. The runner now accepts `--skip_joint_finetune`; the
recommended final command uses it. Joint fine-tuning remains an explicit,
optional ablation rather than an automatic continuation.

## Root cause

1. **Data/observation shift is the dominant cause of poor top-view point
   accuracy.** The deterministic MPJPE changes from 22.406 cm on side IID to
   70.277 cm on top OOD (3.14x). Side and top labels share frame IDs and a
   fixed camera rotation aligns them with about 0.875 mm mean residual, so
   label misalignment is not the explanation. The visible-joint fraction,
   however, changes from 89.6% to 25.0%. After rotation alignment, centered
   point-cloud Chamfer distance is about 40.9 cm and the median top/side
   bounding-box ratios are `[0.765, 0.654, 0.609]`. This is a real visibility
   and observation-geometry shift, not a stale 64-frame cache.

2. **Gaussian uncertainty heads have an algorithmic OOD failure.** Frozen
   independent and graph Gaussian heads become sharper on top OOD even as the
   mean error grows: their mean log-determinant changes are -9.13 and -43.15.
   Side-only joint fine-tuning makes this worse (-121.37 for joint independent),
   consistent with validation-driven scale collapse rather than corrupted
   labels.

3. **Student-t plus graph structure is the useful current response.** Frozen
   graph Student-t widens its uncertainty under the shift (log-determinant
   change +8.64), giving top NLL 2.330 and side/top uncertainty AUROC 0.803.
   This does not prove conditional calibration; it shows that the selected
   heavy-tailed structured family is more informative under this OOD panel.

4. **There is no evidence that compiler lowering caused the failure.** The
   graph precision is the exact 174-coordinate SPD subfamily, the reference
   executor tests passed, and prediction-level metrics recompute from finite
   45-dimensional outputs. A `frame_index` batching issue affected provenance
   metadata only; it did not change prediction order or numeric metrics and is
   recorded in the run audit.

## Next meaningful experiment

The next experiment should change the data protocol once: declared view
augmentation or a mixed-view training split, with validation matched to the
deployment view. Compare that result against frozen graph Student-t. Do not
expand covariance families, add repeated side-only seeds, or resume the
unfinished graph Student-t joint ablation without a new hypothesis.

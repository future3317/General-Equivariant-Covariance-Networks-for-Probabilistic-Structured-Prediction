# ITOP small-sample learning gate

This is a development gate, not a benchmark result. It uses deterministic
prefix-limited geometry caches solely to verify that the existing ITOP training
route can fit signal and execute the same side/top evaluation path as a full
run. The legacy 640-frame smoke cache is a functional test only. Method
selection now uses the separate 1/16 side-train development panel below.

## Fixed scientific contract

- Output: 15 centered 3D joints in the declared `15 x 1o` representation.
- Input: side-view depth reconstructed to 256-point XYZ clouds; no label-based
  centering, augmentation, or split mixing.
- Geometry: exact cached 16-neighbour graph; 640 side-train frames and 256
  frames for each side/top test cache. The cache limit is part of its path and
  metadata, so it cannot be confused with the eventual full-data cache.
- Objective: deterministic MSE first, followed only after that gate by the
  existing graph Student-t proper NLL. The production output semantics and
  graph family are unchanged.
- Precision: BF16 only in the backbone; readout, SPD assembly, Cholesky,
  precision algebra, and likelihood remain FP32. TF32 remains disabled.
- Optimizer: existing AdamW, learning rate `5e-4`, batch size 16, seed 42,
  exact epoch-addressable sampler order, validation early stopping with
  patience 5, and no distributed training.
- Gates: finite loss/gradients, a clear validation-MPJPE reduction over the
  deterministic initial epoch, valid checkpoints/history/side/top artifacts,
  SPD finite graph Student-t loss, and rotation/SPD regression tests. No speed
  claim is made until an eager profile is recorded on the same panel.

## Controlled 1/16 development panel

- Training exposure is the deterministic first 2,487 valid side-train frames
  (approximately 1/16 of the official training split), with its limit encoded
  in both the cache path and run manifest.
- The validation split is drawn only from those 2,487 training frames using
  the recorded seed. Side-test and top-test always use the complete immutable
  test caches; they are never prefix-limited by the development flag.
- This panel selects stable model/training choices only. It must not be cited
  as a benchmark or substituted for the final full side-train experiment.

## Completed development result

The server run `RESULTS/Tpami/ITOP/dev_1of16_20260730/` completed with the
complete 4,863-frame side-test and top-test caches. Graph Student-t was selected
by validation NLL. Its side/top test MPJPE is 28.416/78.545 cm and its proper
NLL is -45.944/75.700, versus 27.929/75.563 cm and -13.326/291.460 for the
independent Gaussian comparison. Top MACE is 0.500 and side/top uncertainty
AUROC is 0.197; these negative calibration diagnostics are retained as part
of the decision record. The result is development evidence only.

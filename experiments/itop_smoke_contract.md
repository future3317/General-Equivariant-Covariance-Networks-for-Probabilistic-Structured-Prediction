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

## Reproducibility and freezing contract

Every controlled ITOP stage records `training_contract_hash` and a companion
`provenance.json`. The record binds the source commit and dirty-tree status,
the complete geometry-cache file hashes, any feature-cache hashes, input
checkpoint hashes, runtime device/dtype/TF32 flags, and the selected compiler
backend. JSON and tensor artifacts are published atomically.

The stage state machine is fixed:

```text
deterministic MSE
  -> freeze deterministic backbone + mean head
  -> train only the uncertainty head with proper Gaussian/Student-t NLL
  -> optionally fine-tune a selected head only when explicitly enabled
```

For `phase=frozen_head`, the code checks and records the exact frozen and
trainable parameter names and counts. The frozen boundary is
`backbone + joint_head.mean_head`; covariance/scale parameters remain
trainable. Thus frozen-head comparisons intentionally have identical means,
MPJPE, and residuals; they compare probabilistic geometry and proper scores,
not point-estimation accuracy. Joint fine-tuning is a separate protocol and
must not be merged with frozen-head results.

The train sampler is seeded as a pure function of `(seed, epoch)`, and worker
Python/NumPy streams are initialized from PyTorch's worker seed. Validation and
test loaders use fixed non-shuffling generators. CUDA TF32 and cuDNN autotuning
are disabled and cuDNN deterministic mode is requested. This fixes the data,
split, optimization, freezing, precision, and artifact semantics; exact
bitwise equality across different CUDA/e3nn kernels is not claimed.

Each completed stage must contain at least:
`best_model.pt`, `last_state.pt`, `history.json`, `metrics.json`,
`predictions_side.pt`, `predictions_top.pt`, `args.json`, `environment.json`,
`compilation.json`, `feature_cache.json`, `provenance.json`, and `train.log`.
The study runner skips a stage only when the complete set is present.

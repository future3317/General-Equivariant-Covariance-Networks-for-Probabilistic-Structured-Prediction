# ITOP 1/16 development panel

This directory contains the small, single-seed development result used to
validate the ITOP compiler path and to decide which graph family to carry into
the final full-data experiment. It is not the final TPAMI benchmark.

- side-train: 2,487 frames (1/16 development panel), seed 42
- side-test: 4,863 complete valid frames
- top-test: 4,863 complete valid frames (cross-view OOD)
- point cloud: 256 points; two interaction layers; one RTX 4090
- output: 15 repeated `1o` vectors (45 coordinates)
- uncertainty: independent Gaussian and graph Student-t; graph Student-t was
  selected by validation proper NLL

The graph Student-t result obtains side NLL `-45.944` and top NLL `75.700`,
compared with `-13.326` and `291.460` for independent Gaussian. MPJPE is not
improved (28.416 cm / 78.545 cm versus 27.929 cm / 75.563 cm), and top MACE
is 0.500. These values are therefore execution and diagnostic evidence, not a
claim of calibrated uncertainty or pose-estimation state of the art.

`metrics.json`, `history.json`, and the prediction files are copied from the
server run after the complete test-cache check. Checkpoints and frozen feature
caches are intentionally not versioned here.

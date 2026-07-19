# Legacy ICML Figure Scripts

This directory contains figure-generation and visualization scripts copied verbatim from the original ICML submission code (`E:/CODE/ICML`). They are preserved for reference and for reproducing ICML 2025 / rebuttal figures, but they have **not been adapted to the new TNNLS architecture** in `gecn/`.

## Files

- `generate_riemannian_diagram_v2.py` — Riemannian manifold / SPD diagram.
- `calibration_combined.py` — Calibration plots (Mahalanobis distance, reliability diagrams).
- `test_and_visualize.py` — General evaluation visualizations (parity plots, equivariance tests, uncertainty calibration).
- `rebuttal_visualization.py` — Runtime, alpha sensitivity, and ablation figures for the rebuttal.
- `alpha_final_correct.py`, `alpha_sensitivity_final.py` — Alpha-sensitivity plots.
- `rebuttal_risk_coverage_analysis.py`, `rebuttal_lambda_max_analysis.py` — Risk-coverage and λmax analysis figures.
- `visualizations/` — Auxiliary icon / crystal structure drawing scripts and generated PNG/PDF outputs.

## Notes

- These scripts still import from the old ICML module layout (`equivariant_network.py`, `dielectric_data_loader.py`, `voigt_utils.py`, etc.). They will not run against the new `gecn/` package without path or import adjustments.
- Generated PNG/PDF outputs are git-ignored at the repository level.

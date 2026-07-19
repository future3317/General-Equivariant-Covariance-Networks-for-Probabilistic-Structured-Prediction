# General Equivariant Covariance Networks for Probabilistic Structured Prediction

Core implementation for the TNNLS manuscript *General Equivariant Covariance Networks for Probabilistic Structured Prediction*.

This repository is being refactored into a clean, modular library (`gecn/`) that separates representation theory, SPD maps, and probabilistic losses. The original ICML conference-code files remain in the repository root for reference but are now considered legacy.

## Architecture

The library is organized into four layers:

1. **Output representations** (`gecn/representations/`)
   - `O3IrrepsSpec`: finite-dimensional orthogonal `O(3)` representations via `e3nn` irreps.
   - `O3SymmetricOperatorBasis`: automatic construction of `Sym²(V)` using `e3nn.o3.ReducedTensorProducts("ij=ji", ...)`.

2. **Equivariant symmetric operators** (`gecn/models/covariance_head.py`)
   - `O3EquivariantSymmetricOperatorHead`: predicts the coefficients of `A(x) ∈ Sym(V)`.
   - `O3EquivariantLowRankCovarianceHead`: predicts a low-rank-plus-isotropic structured parameterization.

3. **SPD maps** (`gecn/spd_maps/`)
   - `MatrixExponentialMap`: `S = exp(A)` (default, bijective).
   - `SpectralSoftplusMap`: spectral softplus with Löwner divided-difference autograd.
   - `SquarePlusIdentityMap`: `S = A² + εI`.
   - `PrecisionExponentialMap`: `S = exp(-B)` (log-precision coordinate).
   - `LowRankPlusIsotropicMap`: `S = σ²I + LLᵀ`.

4. **Probabilistic losses** (`gecn/distributions/`)
   - `GaussianNLL`: proper multivariate Gaussian negative log-likelihood.
   - `StudentTNLL`: proper multivariate Student-t negative log-likelihood with explicit scale/covariance distinction.
   - `RobustSurrogateLoss`: LE-ESO-like robust surrogate, explicitly **not** claimed as a likelihood.

## File guide

| Path | Purpose |
|------|---------|
| `gecn/representations/` | Orthogonal representation specs and `Sym²(V)` basis construction. |
| `gecn/spd_maps/` | Structure-preserving maps from symmetric operators to SPD matrices. |
| `gecn/distributions/` | Gaussian, Student-t, and robust-surrogate losses. |
| `gecn/models/` | Backbone, mean/covariance heads, and `StructuredProbabilisticPredictor`. |
| `gecn/scripts/` | Training scripts for dielectric tensor and elasticity tensor tasks. |
| `gecn/experiments/` | Synthetic covariance-recovery experiment. |
| `tests/` | Unit tests for representations, equivariance, SPD maps, distributions, tensor conversions, synthetic experiment, and integration. |
| `voigt_utils.py` | Voigt / Kelvin-Mandel utilities used by tensor conversions. |
| `matrix_log_transform.py` | Matrix log/exp utilities used by the dielectric pipeline. |
| `atom_features.py` | Atom feature builder used by the data loaders. |
| `dielectric_data_loader.py` | Precomputed-graph dielectric loader wrapped by `gecn/data/dielectric_dataset.py`. |

## Quick start

1. Install in editable mode:
   ```bash
   pip install -e .
   ```

2. Run the test suite:
   ```bash
   python -m pytest tests/ -v
   ```

3. Build a predictor programmatically:
   ```python
   from gecn import (
       O3IrrepsSpec, MatrixExponentialMap, GaussianNLL,
       EquivariantBackbone, EquivariantMeanHead,
       O3EquivariantSymmetricOperatorHead, StructuredProbabilisticPredictor,
   )

   output_spec = O3IrrepsSpec("0e + 2e")  # symmetric rank-2 output
   backbone = EquivariantBackbone(
       hidden_dim=16, lmax=2, num_layers=2,
       atom_feature_dim=49, num_basis=8,
   )
   mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
   cov_head = O3EquivariantSymmetricOperatorHead(
       backbone.irreps_out, output_spec, pool=True,
   )
   model = StructuredProbabilisticPredictor(
       backbone=backbone,
       output_spec=output_spec,
       mean_head=mean_head,
       covariance_head=cov_head,
       spd_map=MatrixExponentialMap(),
       distribution=GaussianNLL(),
   )
   ```

## Training scripts

Three end-to-end scripts are provided under `gecn/scripts/` and `gecn/experiments/`.
They use the new `gecn` architecture and are independent of the legacy root-level
files.

### Dielectric tensor (`0e + 2e` output, full-rank covariance)

```bash
python gecn/scripts/train_dielectric.py \
  --data_dir data/mp_dielectric \
  --save_dir checkpoints_gecn_dielectric \
  --hidden_dim 32 --lmax 2 --num_layers 2 \
  --num_epochs 100 --device cuda
```

The script predicts the dielectric tensor in log-Kelvin-Mandel / `0e + 2e` irrep
space using `MatrixExponentialMap` + `GaussianNLL`, and reports physical-space
MAE by mapping predictions back through the matrix exponential.

### Elasticity tensor (rank-4 output, low-rank covariance)

```bash
python gecn/scripts/train_elasticity.py \
  --data_dir data/mp_elastic \
  --save_dir checkpoints_gecn_elasticity \
  --hidden_dim 48 --lmax 4 --num_layers 2 --rank 8 \
  --num_epochs 100 --device cuda
```

This uses the 21D elasticity-tensor output with a
`LowRankPlusIsotropicMap(rank=8)` covariance, suitable for the high-dimensional
rank-4 target.

### Synthetic covariance recovery

```bash
python gecn/experiments/synthetic_covariance_recovery.py \
  --output_irreps "0e + 2e" \
  --num_train 2000 --num_test 500 \
  --num_epochs 200 --device cuda
```

A controlled experiment where a ground-truth covariance field
`S(x) = exp(A(x))` is generated from a fixed linear map and the model is trained
to recover it. Supported output representations include `"1o"`, `"0e + 2e"`,
and other `O(3)` irreps.

## Notes for reviewers

- The `gecn/` package separates representation theory, SPD maps, and probabilistic losses. It avoids anisotropic eigenvalue jitter, random-noise fallbacks, and improper likelihood claims.
- The code currently provides a full `O3IrrepsSpec` implementation; the abstract `OrthogonalRepresentationSpec` interface leaves room for other compact groups.
- All SPD maps are tested for positive definiteness, finite gradients, and (for the full predictor) `O(3)` equivariance.

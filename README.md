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
| `tests/` | Unit tests for representations, equivariance, SPD maps, distributions, and integration. |
| `equivariant_network.py` | **Legacy** ICML `EquivariantUncertaintyNetwork` (rank-2, 6×6 KM). |
| `train.py` | **Legacy** ICML training script. |
| `stable_loss_implementation.py` | **Legacy** ICML loss with anisotropic eigenvalue jitter. |

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

## Notes for reviewers

- The new `gecn/` package removes the anisotropic eigenvalue jitter, random-noise fallbacks, and "Multivariate Laplace NLL" claims present in the legacy files.
- The code currently provides a full `O3IrrepsSpec` implementation; the abstract `OrthogonalRepresentationSpec` interface leaves room for other compact groups.
- All SPD maps are tested for positive definiteness, finite gradients, and (for the full predictor) `O(3)` equivariance.

# General Equivariant Covariance Networks for Probabilistic Structured Prediction

Core implementation for the TNNLS manuscript *General Equivariant Covariance Networks for Probabilistic Structured Prediction*.

## What this codebase implements

1. **Equivariant SPD distributions on compact group representations**
   - `equivariant_network.py`: `EquivariantUncertaintyNetwork` builds an equivariant symmetric operator `A(X)` and maps it to an SPD covariance via `Σ = exp(A)`.

2. **Automated covariance representation construction**
   - `equivariant_network.py`: uses `e3nn.CartesianTensor("ijkl=jikl=ijlk=klij")` to construct the symmetric rank-4 covariance tensor basis and the Clebsch–Gordan/Cartesian-to-Kelvin–Mandel change of basis.

3. **Spectral SPD parameterizations**
   - `equivariant_network.py` (matrix exponential head).
   - `stable_loss_implementation.py` (eigenvalue-based spectral map for the loss).

4. **Equivariant probabilistic learning**
   - `stable_loss_implementation.py`: stable Log-Euclidean Mahalanobis loss with Gaussian / Laplace options and condition-number control.
   - `train.py`: end-to-end training loop with auxiliary MSE warmup and temperature scaling.

## File guide

| File | Purpose |
|------|---------|
| `equivariant_network.py` | Equivariant message-passing backbone, mean head, and matrix-exponential covariance head. |
| `train.py` | Main training script for dielectric tensor prediction. |
| `dielectric_data_loader.py` | Fast loader for **precomputed** PyG graphs. |
| `dielectric_data_loader_precomputed.py` | On-the-fly graph construction from raw `.pkl` files. |
| `preprocess_edges_full.py` | Preprocessing pipeline that builds and saves precomputed graphs. |
| `stable_loss_implementation.py` | Numerically stable loss: eigen-decomposition, eigenvalue clamping, Laplace/Gaussian NLL, condition-number regularization. |
| `voigt_utils.py` | Conversions between tensor, Voigt, and Kelvin–Mandel representations; equivariant rotation matrices. |
| `matrix_log_transform.py` | Matrix logarithm/exponential transforms for dielectric tensors. |
| `isotropic_utils.py` | Isotropic component handling and denormalization. |
| `atom_features.py` | 49-dimensional atom feature generation. |
| `test_equivariance.py` | Standalone equivariance and baseline tests. |

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare precomputed graphs (run once):
   ```bash
   python preprocess_edges_full.py --data_dir data/mp_dielectric --output_dir data/mp_dielectric
   ```

3. Train:
   ```bash
   python train.py --data_dir data/mp_dielectric --save_dir checkpoints
   ```

4. Test equivariance:
   ```bash
   python test_equivariance.py --checkpoint checkpoints/best.pt
   ```

## Notes for reviewers

- The implementation currently targets **symmetric rank-2 tensor** outputs (e.g., dielectric tensors) under `O(3)`.
- The covariance head is written so that the same spectral-map / Cartesian-tensor machinery generalizes to other compact groups and tensor orders; extending it requires only the appropriate irrep decomposition and `CartesianTensor` symmetry string.
- `stable_loss_implementation.py` still contains an eigenvalue jitter term used for numerical stability during `torch.linalg.eigh`. This is a pragmatic training detail and does not affect the equivariance of the forward covariance construction.

# Representation-Compiled Equivariant Probabilistic Networks

TPAMI submission implementation for *A Representation Compiler for Equivariant
Probabilistic Structured Prediction*.

The main abstraction is a representation compiler. Given a finite-dimensional
orthogonal `O(3)` output representation `V` (or a Cartesian tensor symmetry
formula), it constructs

```text
T(V) = Irreps(V + Sym^2(V))
```

and compiles a target-directed Clebsch--Gordan graph whose final feature space
covers every required angular momentum, parity, and irrep multiplicity. The
same compilation result creates the shared mean/covariance head, covariance
basis, SPD parameterization, and proper Gaussian or Student-t objective.

## What is automatic

- `Sym^2(V)` decomposition and an orthonormal symmetric-operator basis.
- Shortest CG paths from the backbone representation to `T(V)`.
- Parity reachability checks and exact final irrep multiplicities.
- Cartesian tensor symmetries such as `ij=ji` and
  `ijkl=jikl=ijlk=klij`.
- Graph-level (`global`) or element-level (`dense`) output.
- `full`, isotypic `block`, `low_rank`, graph-structured precision, or
  budget-driven `auto` covariance.
- Matrix-exponential full covariance, multiplicity-space block covariance, or
  low-rank-plus-isotropic covariance.
- Proper multivariate Gaussian and Student-t negative log-likelihoods.

For repeated variables with a known graph, the compiler predicts equivariant
unary and relational SPD precision potentials and assembles

```text
Q = BlockDiag(U) + (B kron I)^T BlockDiag(W) (B kron I),  Sigma = Q^{-1}.
```

Training evaluates `logdet(Q)` and `r^T Q r` directly; covariance is formed
only for reporting or sampling. For ITOP this reduces the active uncertainty
output from 1,035 full-covariance parameters to `6 * (15 + 14) = 174` while
retaining a generally dense marginal covariance.

For structured covariance modes, the compiler records both the canonical full
target and the cheaper active target in `compilation.as_dict()`. Approximation
is therefore explicit and auditable.

## Quick start

```python
from models import EquivariantBackbone
from representations import CompilerConfig, O3RepresentationCompiler

backbone = EquivariantBackbone(hidden_dim=32, lmax=2, num_layers=2)

# A symmetric rank-2 Cartesian output. This compiles to V = 0e + 2e,
# Sym^2(V) = 2x0e + 2x2e + 4e, and one quadratic lifting edge.
compiler = O3RepresentationCompiler.from_cartesian(
    "ij=ji",
    CompilerConfig(
        covariance="auto",
        parameter_budget=192,
        output_scope="global",
        objective="gaussian",
    ),
)
compilation = compiler.compile(backbone.irreps_out)
model = compilation.build_model(backbone)

print(compilation.as_dict())
```

For the 21-dimensional elasticity representation, the same API discovers a
canonical full-covariance target up to `8e`. With an `lmax=2` seed this requires
three CG edges. Under the default parameter budget, the active graph selects a
rank-8 covariance instead of materializing all 231 symmetric-operator
parameters.

## Training

```bash
# Rank-2 dielectric output; full covariance is selected explicitly.
python -m scripts.train_dielectric --data_dir data/mp_dielectric --device cuda

# Rank-4 elasticity output; choose auto/full/block/low_rank.
python -m scripts.train_elasticity \
  --data_dir data/mp_elastic \
  --lmax 2 --covariance auto --parameter_budget 192 --rank 8 \
  --objective gaussian --device cuda

# ModelNet40 inertia or shape-covariance target.
python -m scripts.train_modelnet40_inertia \
  --target_type inertia --device cuda

# Download only ITOP depth/label files, compact labels, and skip point clouds.
python -m scripts.download_itop --data_dir data/ITOP --view side

# ITOP standard side-view protocol with graph precision selected at B=192.
python -m scripts.train_itop \
  --data_dir data/ITOP --protocol side --covariance auto \
  --parameter_budget 192 --num_points 1024 --device cuda

# Cross-view OOD protocols use the same entry point.
python -m scripts.train_itop --data_dir data/ITOP --protocol side_to_top --device cuda
```

The ITOP loader reconstructs XYZ points from the documented depth calibration,
centers only by the observable point-cloud centroid, filters invalid frames,
and preserves joint visibility for visible/occluded calibration analysis. It
supports depth noise, point dropout, synthetic occlusion, and 256--2048 point
input budgets without using ground-truth torso centering.

Each training run writes `compilation.json` beside its checkpoint so the exact
representation target, lifting stages, covariance complexity, and objective
are reproducible.

## Layout

| Path | Purpose |
|---|---|
| `representations/compiler.py` | Representation compiler and shared output head |
| `representations/adaptive_lifting.py` | Shortest-path CG graph planning and execution |
| `representations/symmetric_square.py` | `Sym^2(V)` decomposition and covariance basis |
| `representations/graph_structure.py` | Typed repeated-variable output graphs |
| `spd_maps/` | Full, block, low-rank, and graph-precision SPD maps |
| `data/itop_dataset.py` | ITOP depth reconstruction, label compaction, and loaders |
| `evaluation/pose.py` | Pose accuracy, calibration, risk-coverage, and sampling metrics |
| `distributions/` | Proper Gaussian and Student-t objectives |
| `models/` | Equivariant backbone and structured predictor |
| `scripts/` | Reproducible task training entry points |
| `tests/` | Representation, equivariance, SPD, objective, and integration tests |

## Verification

```bash
conda activate EGNN
python -m pytest tests -q -W error
```

The compiler tests cover rank-2 and rank-4 outputs, highest angular momentum,
parity failures, multiplicity coverage, Cartesian round trips, dense/global
heads, complexity selection, equivariance, finite gradients, SPD validity,
graph-precision algebra, and real depth-to-point-cloud data contracts.

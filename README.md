# From Structured Output Types to Predictive Distributions

TPAMI submission implementation for *From Structured Output Types to Predictive
Distributions: A Verified Compiler for Equivariant Learning*.

The normative mathematical and inference semantics are fixed in
[`mathematical_contract.md`](mathematical_contract.md). Code, experiments, and
the manuscript reference this contract rather than maintaining parallel
implicit definitions.

The main abstraction is a verified compiler from structured output declarations
to equivariant predictive distributions. Given a finite-dimensional orthogonal
`O(3)` output representation `V` (or a Cartesian tensor symmetry formula), a
distribution law, and an operator family, it constructs

```text
T(V) = Irreps(V + Sym^2(V))
```

and compiles a target-directed Clebsch--Gordan graph whose final feature space
covers every required angular momentum, parity, and irrep multiplicity. The
compiler output is an executable probabilistic readout, a semantic certificate,
and an execution plan. The same compilation result selects an execution basis,
creates mean and uncertainty-operator representations, an SPD parameterization,
and a proper Gaussian or Student-t objective. The compiler guarantees
representation and probabilistic validity; calibration and the statistical
interpretation of a learned distribution are separate, explicitly evaluated
properties.

The certificate is deliberately scoped: it proves compositionally correct
composition of registered typed primitives and lowerings. It is not a formal
proof of arbitrary user-defined primitives, a completeness theorem for all
equivariant SPD programs, or a calibration/physical-uncertainty guarantee.
The machine-readable boundary is recorded in every compilation report under
`compiler_soundness`; see
[`docs/compiler_certificate_scope.md`](docs/compiler_certificate_scope.md).

The same representation layer can compile `Lambda^2(V)` for orientation
calibration. `EquivariantIsospectralOrientationCalibrator` predicts a skew
generator `K` in an orthonormal basis and applies `exp(K) S exp(K)^T`. It is
zero-initialized and preserves SPD, eigenvalues, log-volume, and condition
number. Dense orientation calibration is intended only for small outputs;
high-multiplicity and graph-structured outputs must select an explicit
block/low-rank policy with its compiled cost recorded.

For heteroscedastic full covariance, `SpectralWindowCovariance(a, b)` compiles
`Q diag(exp(a + (b-a) sigmoid(lambda))) Qᵀ`. This map is orthogonally
conjugation-equivariant, strictly SPD, and bounds every covariance eigenvalue
in `[exp(a), exp(b)]` on the same path used for training, evaluation, and
inference. Gaussian NLL therefore remains the proper log score for this
explicit restricted distribution family; it is not an inference-only clamp.

## Scope and current status

The compiler contract is group-agnostic for finite-dimensional orthogonal output
representations. The released executable backend is currently validated for
orthonormal `O(3)` contracts using e3nn-real layouts; extending the semantic IR
to another compact orthogonal group does not imply that a numerical backend for
that group is already shipped. This distinction is intentional: the compiler
can reject an unreachable or unsupported lowering with a certificate instead of
silently changing the requested representation or probabilistic family.

The repository contains the complete ITOP data interface, geometry/feature
cache builders, single-GPU study runner, and metric/evaluation code. A controlled
one-seed 1/16 server development panel exercises the complete side/top OOD path
on 2,487 side-train frames and all 4,863 frames in each test view. Its artifacts
are archived with the TPAMI manuscript. It is development evidence for method
selection and compiler execution, not a replacement for the final full-data
benchmark.

The panel selects graph Student-t by validation proper NLL. On side/top test it
obtains MPJPE 28.416/78.545 cm and NLL -45.944/75.700, while independent
Gaussian obtains 27.929/75.563 cm and -13.326/291.460. Top-view MACE remains
0.500 and the uncertainty AUROC is 0.197, so no calibration or SOTA claim is
made. See `results/itop_dev_1of16_20260730/` and
`figures/itop_dev_1of16_20260730/` for the checked-in records and figures.

The completed full-data single-seed audit uses all 17,991 valid side-train
frames and 512 points. Frozen graph Student-t is selected as the main head:
side/top MPJPE is 22.406/70.277 cm, side/top NLL is -55.955/2.330, and the
side-versus-top uncertainty AUROC is 0.803. Frozen Gaussian heads become
overconfident on the top-view shift, while side-only joint fine-tuning improves
top MPJPE slightly but collapses top NLL to 4,988.677. These results support a
cross-view OOD diagnostic, not a calibration or SOTA claim. The full root-cause
record is [`docs/itop_final_root_cause_diagnosis.md`](docs/itop_final_root_cause_diagnosis.md).

## Evidence and uncertainty semantics

The current learned result is one single-seed, single-GPU Materials Project
run on an RTX 4090. It uses the strict `mean -> covariance -> joint` state
machine, hidden width 32, `lmax=2`, two interactions, batch size 128, BF16
backbone autocast, FP32 operator/NLL algebra, centered spectral control, and a
Student-t objective with `nu=5`.

Server checkpoint and audit directory:

```text
/home/workspace/lrh/RESULTS/Tpami/dielectric/unified_student_t_centered_b128_20260726/
```

The copied local audit is
`experiments/results/unified_student_t_centered_b128_20260726/joint/`. Final
test metrics are NLL `-2.6248`, physical MAE `2.0611`, log-Kelvin--Mandel MAE
`0.1604`, Energy Score `0.4427`, sliced CRPS `0.1551`, and risk--coverage AUC
`0.5623`. Joint ellipsoid coverage is `43.8/59.4/71.2/76.9%` at nominal
`50/80/90/95%`; ACE is `0.1025`. The covariance basis and local Jacobian both
have rank 21, while the angular whitening defect is `10.4612` and the maximum
condition number is `54.5975` (the certified `exp(4)` bound). All metrics and
figures use the recorded inference-contract hash.

This artifact demonstrates a valid, expressive full-scatter execution path;
it does **not** identify a material's physical aleatoric covariance. With one
fixed-protocol DFPT label per structure, a learned distribution mixes surrogate
error, finite-data/model uncertainty, and any irreducible observation noise.
It is reported as a predictive-distribution audit, including its undercoverage
and angular-shape failure, rather than as recovered physical uncertainty.

Validation-only scalar temperature calibration is intentionally **not applied**:
it worsens held-out test NLL from `-2.6248` to `-2.4257` (block calibration:
`-2.3850`). Final figures are in
`experiments/results/unified_student_t_centered_b128_20260726/joint/figures_fixed/`.

## What is automatic

- Analytic `Sym^2(V)` irrep decomposition (including high-multiplicity outputs)
  with lazy dense basis materialization only when a dense operator is selected.
- Shortest CG paths and a depth-minimal, target-pruned representation frontier
  from the backbone representation to `T(V)`.
- Parity reachability checks and exact final irrep multiplicities.
- Automatic exact STF-coordinate/dense-projector lowering for one-edge rank-2
  full covariance and complete `spherical_cg` execution for other targets.
- Cartesian tensor symmetries such as `ij=ji` and
  `ijkl=jikl=ijlk=klij`.
- Graph-level (`global`) or element-level (`dense`) output.
- `full`, isotypic `block`, `low_rank`, graph-structured precision, or
  budget-driven `auto` covariance.
- Matrix-exponential full covariance, multiplicity-space block covariance, or
  low-rank-plus-isotropic covariance.
- Proper multivariate Gaussian and Student-t negative log-likelihoods.
- Symmetric whitening, Student-t radial PIT, angular whitening defect, and
  irrep-resolved calibration diagnostics.
- Gaussian/Student-t Energy Score and isotropic sliced CRPS for
  orientation-sensitive proper-score evaluation.
- Centered spectral windows that control log-volume separately from trace-free
  log-shape.

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
target and the cheaper active target in a versioned compilation report. It
separately records whether the covariance family restricts the canonical full
target and whether backend lowering is algebraically exact. A budget-selected
low-rank or graph family can therefore never be mislabeled as full covariance,
and contraction truncation can never be mislabeled as checkpoint-equivalent.

## Optional uncertainty-learning protocols

The dielectric trainer now exposes `--faithful_joint`, which sends MSE-only
gradients through the mean/shared trunk and uses detached residuals for the
covariance path. `scripts/build_dielectric_oof_residuals.py` constructs fixed
out-of-fold residual caches from mean checkpoints, and
`--oof_residuals <cache.pt>` consumes them during covariance training. The
`evaluation.ensemble` API and `scripts/evaluate_dielectric_ensemble.py` report
the law-of-total-covariance decomposition, exact finite-mixture NLL, Energy
Score, and dependence-sensitive variogram score. These extensions have not yet
been used to claim improved dielectric performance from an ensemble.

The completed controlled ordinary-versus-OOF faithful comparison was a
**No-Go**: OOF slightly worsened NLL, angular whitening defect, Energy Score,
sliced CRPS, and risk--coverage while leaving coverage unchanged; its tiny
variogram change was not decision-relevant. OOF and pseudo-Wasserstein are
therefore not expanded. The next independent protocol is a finite deep
ensemble, followed by split-conformal calibration of prediction regions.
Moment covariance is
`mean(Sigma_m) + Cov(mu_m)`, where each Student-t member contributes
`Sigma_m = nu/(nu-2) * S_m`; it is not relabeled as a single Student-t density.

The uncertainty wrappers are deliberately outside the compiler core:

- Proper Gaussian/Student-t NLL trains a predictive distribution in the
  compiler-selected output coordinates.
- A finite ensemble is an equal-weight mixture with mixture NLL; member spread
  is an epistemic diagnostic, not a single Student-t scale.
- Split conformal uses a held-out calibration split and a compiler-produced SPD
  shape `C(x)` to output equivariant ellipsoidal regions. It guarantees only
  finite-sample marginal coverage under exchangeability, never pointwise
  conditional coverage.
- Repeated-protocol labels for the *same* structure are required before a full
  directional covariance can be interpreted as DFPT protocol uncertainty.

The executable wrapper is `evaluation.conformal.fit_split_conformal`. It takes
calibration means, SPD shapes, and targets, returns the finite-sample order
statistic and threshold, and evaluates membership and region log-volume without
changing the probabilistic meaning of the compiler output. A reproducible
Student-t shape audit is available with:

```bash
python -m experiments.synthetic_conformal_audit \
  --output results/synthetic_conformal_audit.json \
  --calibration-size 255 --test-size 10000 --alpha 0.1
```

An optional, explicitly non-directional auxiliary is also available for
audited ablations. `scripts/build_dielectric_pseudo_covariance.py` consumes a
five-fold, train-only OOF cache and constructs an isotropic residual-covariance
target with invariant structural kNN, negative squared-distance weights, and
recorded shrinkage/provenance. It does not construct a full directional target:
without a verified neighbour-to-query O(3) transport certificate the runtime
rejects that mode. `covariance_warmup` sends its Wasserstein gradient only to
the covariance projection, then a normal covariance and faithful Student-t
stage must follow. It is not a replacement for the proper NLL, OOF residuals,
or an ensemble, and no dielectric result currently claims a benefit from it.

Every operator program is verified in a typed environment containing its
named representation bindings, declared output representation, and optional
output graph. The verifier checks parameter slices after layout conversion,
`Sym^2(V)` and `rV` leaves, trivial-scalar primitives, multiplicity blocks,
and graph pullback domains before executable lowering. Optimized full,
low-rank, block, and graph maps are enabled only after exact whole-program
template identity; the report records the template hash, binding
correspondence, layout transform, graph identity, and rank. Near-miss programs
use the generic recursive interpreter.

For `LowRankCovariance(r)`, the isotropic term is `softplus(s) I` with no fixed
positive floor. It is therefore strictly SPD for every finite parameter and,
when `r >= dim(V)`, represents the same full SPD family as the unrestricted
parameterization. For `r < dim(V)` it remains a documented strict subfamily.

## Stable staged interface

The interface separates output semantics, feature reachability, lowering, and
module binding. The governing contract is:

```text
V determines what must be predicted; H^(0) determines whether and how it can be computed.
```

`describe_output()` therefore accepts `V` alone but is explicitly
non-executable. It returns `V`, `Sym^2(V)`, `T(V)`, dimensions, multiplicities,
parity, and highest angular momentum while marking reachability as unknown:

```python
from equivcompiler import describe_output

semantics = describe_output("ijkl=jikl=ijlk=klij")
assert semantics.executable is False
assert semantics.reachability == "unknown_without_seed"
```

Executable compilation uses one of two unambiguous entry points. A standalone
readout receives a complete `FeatureSpec`; a full predictor receives a
backbone. Both call the same pure `plan_readout()` internally:

```python
from equivcompiler import (
    AutoBudget,
    ExactExecutorCandidates,
    ExactOnly,
    FeatureSpec,
    FullCovariance,
    LowRankCovariance,
    PreferExecutor,
    compile_readout,
)

seed = FeatureSpec.from_irreps(
    "32x0e + 16x1o + 16x2e",
    group="O3",
    scope="global",
    layout="e3nn",
    basis_convention="e3nn_real_v1",
)
readout, report = compile_readout(
    seed,
    output="ijkl=jikl=ijlk=klij",
    covariance=AutoBudget(
        max_parameters=192,
        candidates=(FullCovariance(), LowRankCovariance(8)),
    ),
    fidelity=ExactOnly(),
    executor=ExactExecutorCandidates(),
    cost=PreferExecutor(("spherical_cg", "cartesian_stf")),
    distribution="student_t",
)
```

`FeatureSpec` fixes group, scope, declared irrep order and multiplicity, storage
layout, real-basis convention, parity convention, and pooling permission. Its
SHA-256 fingerprint is part of the compatibility hash. Binding a plan to a
backbone with a different contract fails before checkpoint loading.

Advanced users can dry-run without constructing a neural module:

```python
from equivcompiler import FullCovariance, plan_readout

plan = plan_readout(
    seed,
    output="ij=ji",
    covariance=FullCovariance(),
    fidelity=ExactOnly(),
    # Optional native cuEquivariance lowering (Linux/WSL + fused ops wheel).
    lifting_backend="cueq",
    cueq_method="fused_tp",
)
plan.report.save("compilation.json")
readout = plan.build_readout(device="cuda")
```

`lifting_backend` is a compiler lowering choice and is independent of the
backbone's message-passing `tp_backend`. Both choices are included in the
execution/compatibility signature. cuEquivariance accelerates Clebsch--Gordan
tensor products only; it does not replace SPD assembly, matrix exponential or
Cholesky, log-determinant/Mahalanobis evaluation, or the proper likelihood.
The fused method requires `cuequivariance_ops_torch` (installed by the WSL and
server lock files). If it is unavailable, `cueq_method="fused_tp"` fails
explicitly; the compiler never silently falls back to another kernel.

The report includes:

- decompositions of `V`, `Sym^2(V)`, the canonical target, and active target,
  including highest angular momentum, parity, and every multiplicity;
- a shortest-CG-path reachability proof and zero-deficit multiplicity check;
- structured failure certificates for unreachable parity, incompatible
  backends, missing seed types, and unsupported parameterizations;
- orthogonal canonical/active reachability, covariance-family coverage, and
  execution-lowering exactness;
- emitted-coordinate, readout, and executable parameter counts plus
  covariance-specific storage and likelihood complexity;
- exact covariance/scatter/precision semantics and the selected proper
  likelihood.

The public policies are deliberately orthogonal. `FullCovariance()`,
`LowRankCovariance(rank)`, `IsotypicBlockCovariance()`, and
`GraphPrecision(graph)` define operator families. `AutoBudget(max_parameters,
candidates)` minimizes emitted parameter count among feasible candidates, while
`FirstFeasible(max_parameters, priority)` follows an explicit user order.
`ExactOnly()` and `TruncatedMultiplicityRank(rank)` control fidelity;
`ExactExecutorCandidates()` or `SpecificExecutor(name)` control executor
eligibility; and `PreferExecutor(priority)` or `MinimizeLatency(...)` controls
cost selection. Low-rank and graph models normally report `strict_subfamily`
plus `exact_for_active_family`; only explicit contraction truncation reports
`approximate_for_active_family`. `ExactOnly()` never selects truncation.
The selected active representation is the compilation gate. The unrestricted
full-family target remains a diagnostic reference when a different structured
parameterization is selected, so an unreachable canonical reference cannot
silently change the requested statistical family.

This separation is also a safeguard. Coordinate-wise Cholesky is not exposed
because it is not conjugation equivariant; a Gaunt-only shortcut is not
accepted as a general backend because it can miss parity channels and irrep
copies. Low-level invalid requests and reachability failures carry
machine-readable certificates and possible remedies.

For `V = 0e + 2e`, the specialized operator basis uses the bijection
`A <-> (a, b, P, Q, H)` with dimensions `1 + 1 + 5 + 5 + 9 = 21`. Frozen
`2 x 2 -> 0, 2, 4` projectors are generated once in e3nn's orthonormal real
irrep coordinates. The exact compiler backend preserves the complete lifting
stage, including mixed-parity input paths, and retains the identical flat
`FullyConnectedTensorProduct` weights, normalization, module names, and output
basis. It changes only the multiplicity/angular contraction schedule, so an
existing spherical checkpoint loads with `strict=True`. A requested
`stf_contraction_rank` below a path's matrix rank creates a separate,
explicitly approximate CP parameterization.

The lifting certificate is minimal in tensor-product depth. Its retained irrep
frontiers are the union of selected shortest paths. Each executable stage
remains a fully connected tensor product over those types; the implementation
does not claim globally minimal frontier width, instruction count, FLOPs, or
parameterization.

## Algebra-preserving performance paths

- The exact rank-2 lowering replaces each retained one-edge spherical CG
  tensor product with multiplicity-first dense projectors. Strict checkpoint
  loading, mapped output, shared-feature/weight/loss gradients, rotations, and
  reflections are covered by regression tests. Speed is treated as
  hardware-, width-, and execution-mode-dependent rather than guaranteed.
- Message-passing degree normalization is computed once per backbone forward
  and reused by every layer; native `index_add_` performs the sum aggregation.
- Proper objectives call `SPDMap.statistics()`. Low-rank and graph backends
  reuse one factorization/local-SPD construction for log determinant and
  Mahalanobis terms.
- Tree output graphs such as ITOP use exact block Schur elimination for
  `logdet(Q)`; cyclic graphs retain dense Cholesky.
- `--compile_tp` compiles only edge tensor products. It fails explicitly when
  CUDA compilation dependencies such as Triton are unavailable and never
  silently changes backend.
- Every training/profile entry point exposes the same
  `--tp_backend/--cueq_method/--compile_tp` contract.
  `--tp_backend cueq --cueq_method naive --compile_tp` uses cuEquivariance's
  e3nn-compatible `mul_ir` layout through the Windows Triton path. On WSL,
  `--cueq_method fused_tp` selects NVIDIA's native CUDA kernels. Missing fused
  ops raise immediately instead of silently changing the requested method.
- The staged compiler also exposes the lifting tensor-product backend as an
  explicit lowering choice. Pass `lifting_backend="cueq"` and
  `cueq_method="fused_tp"` to `plan_readout` on Linux/WSL when the native ops
  wheel is installed; `"naive"` is the portable eager path. This choice is
  included in the compatibility hash and report, and has an exact e3nn-layout
  regression test. It is independent of the Cartesian-STF operator executor.
- Compare the three tensor-product kernels on the actual server GPU with
  `python scripts/benchmark_tp_backends.py`; the script records numerical
  error, forward/forward-backward latency, and the complete shape contract.
- Dielectric graphs can be converted to cache-friendly shards with
  `python -m scripts.shard_dielectric_graphs`. Use
  `--dataset_storage shards` to enable shard-aware batching.

These paths preserve the mathematical model. Compiled kernels and native
scatter reductions can reorder floating-point operations, so equivalence is
tested with numerical tolerances rather than claimed bitwise identity.

### Measured cuEquivariance status

The checked-in benchmark record
[`audit_results/cuequivariance_tp_benchmark_4090.json`](audit_results/cuequivariance_tp_benchmark_4090.json)
was produced on one NVIDIA RTX 4090 (CUDA 12.6, PyTorch 2.6). For the recorded
`3x0e+2x1o` by `1x0e+1x1o` tensor product, the fused kernel was about `2.20x`
faster in forward and `1.41x` faster in forward/backward than the e3nn eager
reference, with maximum absolute output error below `2e-6`. The naive
cuEquivariance path is retained as a portable correctness path and was slower
for this configuration; performance is not assumed to transfer across widths
or GPUs.

The latest isolated server regression run completed with `217 passed, 12
skipped`. Skips are dependency/device-gated tests (for example, fused CUDA
tests on a CPU-only checkout), not silently selected fallbacks. No training was
started as part of this validation.

## Quick start

On the Windows workstation, the existing `EGNN` environment is configured
once with the shared data root:

```powershell
conda env config vars set EQUIVCOMPILER_DATA_ROOT=E:\DATA\Tpami -n EGNN
conda activate EGNN
```

The tested fused-kernel environment is WSL2 with Python 3.11 and CUDA 12.8:

```bash
micromamba env config vars set \
  EQUIVCOMPILER_DATA_ROOT=/mnt/e/DATA/Tpami -n equivcompiler
micromamba activate equivcompiler
python -m pip install -r requirements-wsl.txt
```

For the lab server with NVIDIA driver 535, use the CUDA 12.6 lock. NVIDIA's
CUDA 12.x minor-version compatibility allows this on the 535 driver, and
cuEquivariance 0.10 requires cuBLAS 12.5 or newer (so a cu121 PyTorch wheel is
not compatible):

```bash
conda create -n equivcompiler python=3.11 pip -y
conda env config vars set \
  EQUIVCOMPILER_DATA_ROOT=/home/workspace/lrh/DATA/Tpami -n equivcompiler
conda activate equivcompiler
python -m pip install -r requirements-server.txt
```

Dataset payloads live outside the Git checkout. Configure their root once per
environment through `EQUIVCOMPILER_DATA_ROOT`; loaders and CLI scripts then
select their dataset-specific subdirectory automatically. The current roots
are:

```text
Windows: E:\DATA\Tpami
WSL:     /mnt/e/DATA/Tpami
Server:  /home/workspace/lrh/DATA/Tpami
```

The root must contain `ITOP`, `modelnet40`, `mp_dielectric`, and `mp_elastic`.
`--data_dir` and `--cache_path` remain available only for intentional
per-command overrides. Python loader code stays in the repository's `data/`
package; no dataset symlinks are required.

```python
from equivcompiler import ExactOnly, FullCovariance, compile_predictor
from models import EquivariantBackbone

backbone = EquivariantBackbone(hidden_dim=32, lmax=2, num_layers=2)
model, report = compile_predictor(
    backbone,
    output="ij=ji",
    covariance=FullCovariance(),
    fidelity=ExactOnly(),
    distribution="gaussian",
)
```

For the 21-dimensional elasticity representation, the same API discovers a
canonical full-covariance target up to `8e`. With an `lmax=2` seed this requires
three CG edges. With an explicit `AutoBudget(max_parameters=192,
candidates=(FullCovariance(), LowRankCovariance(8)))` policy, the compiler
selects the 169-parameter rank-8 subfamily instead of materializing all 231
symmetric-operator parameters and records that restriction in the report.

## Training

```bash
# Dielectric uses one immutable RunSpec across training, evaluation,
# calibration, and figure generation.  Train the deterministic mean first.
python -m scripts.train_dielectric --training_stage mean \
  --save_dir /path/to/dielectric_mean --device cuda

# Cross-fitted means: train exactly five models. Fold k excludes the samples
# with deterministic fold id k; all arguments below must agree across folds.
for k in 0 1 2 3 4; do
  python -m scripts.train_dielectric --training_stage mean \
    --oof_folds 5 --oof_holdout_fold "$k" --oof_seed 0 \
    --save_dir "/path/to/oof_mean/fold$k" --device cuda
done
python -m scripts.build_dielectric_oof_residuals \
  --checkpoint_dirs /path/to/oof_mean/fold0 /path/to/oof_mean/fold1 \
    /path/to/oof_mean/fold2 /path/to/oof_mean/fold3 /path/to/oof_mean/fold4 \
  --folds 5 --seed 0 --output /path/to/oof_residuals.pt --device cuda

# Use the fixed OOF cache only for covariance/joint stages. The builder checks
# the exact fold/seed contract and refuses in-sample or mismatched checkpoints.
python -m scripts.train_dielectric --training_stage covariance \
  --init_checkpoint /path/to/dielectric_mean --oof_residuals /path/to/oof_residuals.pt \
  --save_dir /path/to/oof_covariance --device cuda

# Fit only the covariance projection from that exact mean checkpoint.
python -m scripts.train_dielectric --training_stage covariance \
  --init_checkpoint /path/to/dielectric_mean \
  --save_dir /path/to/dielectric_covariance --device cuda

# Optional only after all pseudo-label Go/No-Go tests pass: warm up the
# covariance projection from a train-only cache, then continue with the
# ordinary covariance stage above (which accepts this checkpoint).
python -m scripts.build_dielectric_pseudo_covariance \
  --oof_residuals /path/to/oof_residuals.pt --output /path/to/pseudo.pt \
  --k 32 --tau 1.0
python -m scripts.train_dielectric --training_stage covariance_warmup \
  --init_checkpoint /path/to/dielectric_mean \
  --pseudo_covariance_cache /path/to/pseudo.pt \
  --save_dir /path/to/dielectric_warmup --device cuda

# Jointly fine-tune from the covariance checkpoint (use a smaller explicit LR).
python -m scripts.train_dielectric --training_stage joint --lr 1e-4 \
  --init_checkpoint /path/to/dielectric_covariance \
  --save_dir /path/to/dielectric_joint --device cuda

# Optional one-time I/O conversion. Shard-aware batching preserves shuffle
# while avoiding random per-graph file opens.
python -m scripts.shard_dielectric_graphs \
  --shard_size 256
python -m scripts.train_dielectric \
  --dataset_storage shards \
  --shard_cache_size 2 --num_workers 4 --persistent_workers --device cuda

# Windows cuEquivariance frontend compiled through Triton.
python -m scripts.train_dielectric \
  --tp_backend cueq \
  --cueq_method naive --compile_tp --device cuda

# WSL native cuEquivariance CUDA kernels (no torch.compile required).
python -m scripts.train_dielectric \
  --tp_backend cueq \
  --cueq_method fused_tp --device cuda

# Re-evaluate an existing dielectric checkpoint.  The JSON output contains
# proper-likelihood-space calibration/ellipsoid coverage, sharpness, whitened
# residual diagnostics, and spectral-window bound utilization.  Calibration is
# computed in log--Kelvin--Mandel space (the Gaussian likelihood's coordinate
# system); covariance matrices are materialized in FP64 for this audit so
# floating-point reconstruction error is not reported as a spectral violation.
python -m scripts.train_dielectric \
  --save_dir /path/to/dielectric_run --evaluate_only --device cuda

# Legacy runs are made compatible only through this explicit, audited migration.
python -m scripts.migrate_dielectric_run_spec \
  --checkpoint_dir /path/to/legacy_dielectric_run

# Rank-4 elasticity output; choose auto/full/block/low_rank.
python -m scripts.train_elasticity \
  --lmax 2 --covariance auto --parameter_budget 192 --rank 8 \
  --objective gaussian --device cuda

# ModelNet40 inertia or shape-covariance target.
python -m scripts.train_modelnet40_inertia \
  --target_type inertia --device cuda

# One-time, non-destructive ModelNet40 scale cleaning. The rule is fitted on
# centered point-cloud radii from the training inputs only; it writes a new
# cache and JSON audit without overwriting the raw cache. The cleaned cache is
# the default used by training.
python -m scripts.clean_modelnet40_cache

# One-time exact k-NN precomputation for the standard training geometry.
python -m scripts.precompute_modelnet40_graphs \
  --num_points 1024 \
  --num_neighbors 16

# Download only ITOP depth/label files, compact labels, and skip point clouds.
python -m scripts.download_itop --view side

# Dry-run the one-seed development schedule.  It exposes exactly one physical
# GPU to all children and never launches DDP.  The 1/16 panel limits only
# side-train (2,487 frames); side-test and top-test remain complete.
python -m scripts.run_itop_study \
  --data_dir /home/workspace/lrh/DATA/Tpami/ITOP \
  --study_dir /home/workspace/lrh/RESULTS/Tpami/ITOP \
  --profile development --gpu 3 --seeds 42 \
  --train_cache_sample_limit 2487 --dry_run

# Remove --dry_run to execute. The final protocol uses 512 points and one
# explicitly recorded seed on the same single GPU. Completed stages are skipped,
# interrupted non-joint stages resume from last_state.pt, and patience is five
# validation epochs. Joint uncertainty-head fine-tuning is an optional ablation;
# skip it for the selected final protocol.
python -m scripts.run_itop_study \
  --data_dir /home/workspace/lrh/DATA/Tpami/ITOP \
  --study_dir /home/workspace/lrh/RESULTS/Tpami/ITOP \
  --profile final --gpu 3 --skip_joint_finetune
```

The ITOP loader reconstructs XYZ points from the documented depth calibration,
centers only by the observable point-cloud centroid, filters invalid frames,
and preserves joint visibility for visible/occluded calibration analysis. The
controlled study trains a deterministic model, caches its frozen pooled
features, then compares independent-joint Gaussian, graph Gaussian, and graph
Student-t operators behind the same frozen backbone and deterministic mean
readout. It reports side-view IID and side-to-top OOD accuracy, likelihood,
calibration, risk--coverage, occlusion, and residual-correlation metrics. The
default final decision uses the frozen-head comparison; joint fine-tuning is an
optional ablation because the completed side-only joint run harmed cross-view
OOD likelihood. The
development/final point budgets are 256/512; the development subset is only a
configuration gate, while the final protocol uses all side-train frames. No
ground-truth torso centering is used.

Training writes an atomic `history.json` after every epoch with train/validation
losses, proper-NLL components, learning rates, and gradient norms. Non-finite
targets, losses, components, gradients, and early-stopping criteria fail
immediately. `last_state.pt` stores model/optimizer/scheduler state, early-stop
state, history, and Python/NumPy/Torch/CUDA RNG streams; the shuffled sample
order is an explicit function of seed and epoch, so an interrupted cached-data
study resumes at the next epoch without changing its sample order.
Training and frozen-feature extraction require the immutable geometry caches;
there is no online raw-HDF5 fallback. Raw depth and compact-label files are read
only by the explicit geometry-precomputation stage and must use the canonical
direct-file layout. `--continue_run` likewise requires a schema-v3
`last_state.pt`; an old checkpoint or an incomplete run directory without that
state is rejected instead of restarted implicitly.

Each training run writes `compilation.json` beside its checkpoint so the exact
representation target, lifting stages, covariance complexity, execution
backend/exactness, and objective are reproducible.

The backend microbenchmark is synthetic and never starts training:

```bash
python -m scripts.benchmark_stf_backend \
  --device cuda --batch-sizes 32,128,256 \
  --multiplicities 8,16,32,64 \
  --dtypes float32 --executions eager,compile \
  --warmup 20 --repeats 100 \
  --output results/stf_backend_sweep_cuda.json
```

The compiler-level system closure is measured separately from training. It
records plan/materialization time, synchronized readout forward and
forward/backward latency, peak CUDA memory, batch/multiplicity scaling, FP32
or BF16, and eager/`torch.compile` status. Unsupported combinations are
written as explicit errors and never silently fall back:

```bash
python -m scripts.benchmark_compiler_system \
  --device cuda --warmup 10 --repeats 100 \
  --output results/compiler_system_cuda.json
```

The statistical closure is a controlled ground-truth benchmark rather than a
claim about physical aleatoric covariance. A compiler-produced teacher emits
known Student-t scatter for full, low-rank, isotypic-block, and graph-precision
families; repeated observations test recovery, Student-t radial coverage,
equivariance, and arbitrary orthogonal coordinate-change invariance:

```bash
python -m experiments.synthetic_covariance_benchmark \
  --device cuda --contexts 128 --replicates 32 --steps 500 --seeds 0,1,2 \
  --output results/synthetic_covariance_benchmark_cuda.json
```

For the same rank-2 output type, the cross-family mode varies only the declared
teacher/learner operator family and writes a $3\times3$ recovery matrix.  It
records whether active-target compilation selected a shared lifting trunk; a
non-shared cross-family row is not silently treated as a head-only ablation:

```bash
python -m experiments.synthetic_covariance_benchmark \
  --cross-family-matrix --families full,low_rank,isotypic_block \
  --device cuda --contexts 128 --replicates 32 --steps 500 --seeds 0,1,2 \
  --output results/synthetic_cross_family_covariance_recovery_cuda.json
```

An existing ModelNet40 compiler checkpoint can be audited without training:

```bash
python -m scripts.benchmark_checkpoint_lowering \
  --checkpoint runs/modelnet40/inertia_clean_seed42/best_model.pt \
  --tp-backend cueq --cueq-method fused_tp --device cuda \
  --batch-size 16 --warmup 10 --repeats 30 --no-tf32 \
  --output results/checkpoint_lowering_modelnet40_cuda.json
```

Exact checkpoint migration is a formal, no-retraining workflow. The tool
strictly instantiates both compilations, verifies learned names and shapes,
loads both heads, performs a deterministic numerical equivalence check, removes
only regenerable e3nn code-generation buffers, and writes a SHA-256 audit
sidecar. It refuses incompatible, approximate, ambiguous, and in-place
conversions:

```bash
equiv-compiler convert-checkpoint \
  --checkpoint runs/modelnet40/inertia_clean_seed42/best_model.pt \
  --destination runs/modelnet40/inertia_clean_seed42/best_model_stf.pt \
  --from spherical-cg --to cartesian-stf \
  --output-representation "0e + 2e" \
  --seed-irreps "32x0e + 16x1o + 16x2e" \
  --feature-scope node --covariance full --output-scope global
```

## Layout

| Path | Purpose |
|---|---|
| `representations/compiler.py` | Representation compiler and shared output head |
| `representations/report.py` | Versioned reachability, semantics, complexity, and approximation report |
| `representations/diagnostics.py` | Machine-readable compilation and safeguard certificates |
| `representations/adaptive_lifting.py` | Shortest-path CG graph planning and execution |
| `representations/symmetric_square.py` | `Sym^2(V)` decomposition and covariance basis |
| `representations/cartesian_stf.py` | Rank-2 STF-coordinate operator basis and specialized tensor square |
| `representations/dense_projector.py` | Checkpoint-preserving multiplicity-first lowering of compiled CG stages |
| `representations/graph_structure.py` | Typed repeated-variable output graphs |
| `representations/operator_ir.py` | Typed operator IR and binding-aware soundness verifier |
| `representations/operator_lowering.py` | Recursive lowering and exact-template optimization certificates |
| `spd_maps/` | Full, block, low-rank, and graph-precision SPD maps |
| `data/itop_dataset.py` | ITOP depth reconstruction, label compaction, and loaders |
| `evaluation/pose.py` | Pose accuracy, calibration, risk-coverage, and sampling metrics |
| `distributions/` | Proper Gaussian and Student-t objectives |
| `equivcompiler/api.py` | Stable declarative Python interface |
| `equivcompiler/specs.py` | Output semantics and feature contracts/fingerprints |
| `equivcompiler/policies.py` | Independent covariance-family and lowering policies |
| `equivcompiler/planning.py` | Immutable dry-run planning and compatibility hashes |
| `equivcompiler/modules.py` | Deferred executable readout materialization |
| `equivcompiler/checkpoint.py` | Strict exact checkpoint migration and audit |
| `models/` | Equivariant backbone and structured predictor |
| `scripts/` | Reproducible task training entry points |
| `tests/` | Representation, equivariance, SPD, objective, and integration tests |

## Verification

```bash
# Local CPU/EGNN checks (cuEquivariance and fused CUDA tests are skipped when
# their optional dependencies or a CUDA device are absent).
conda activate EGNN
python -m pytest tests -q

# The authoritative CUDA regression is run on the lab server in the
# `equivcompiler` environment, with one physical GPU exposed to the process.
conda activate equivcompiler
CUDA_VISIBLE_DEVICES=0 python -m pytest tests -q

# Reproduce the tensor-product measurement recorded above.
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_tp_backends.py \
  --output audit_results/cuequivariance_tp_benchmark_4090.json
```

The compiler tests cover rank-2 and rank-4 outputs, highest angular momentum,
parity failures, multiplicity coverage, Cartesian round trips, dense/global
heads, complexity selection, equivariance, finite gradients, SPD validity,
CG/STF output and gradient equivalence, projector orthogonality, explicit rank
truncation, graph-precision algebra, and real depth-to-point-cloud data contracts.
The suite also covers typed binding failures and optimization near-misses.
GitHub Actions runs the CPU suite and Ruff without requiring cuEquivariance,
Triton, or CUDA.

# Independent Teacher Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace circular compiler-generated teacher evidence with a three-seed controlled recovery benchmark whose complete data-generating path is an independent NumPy/SciPy oracle.

**Architecture:** A pure NumPy/SciPy oracle writes immutable `.npz` datasets and JSON manifests before any learner exists. The existing benchmark remains responsible for compiler learner construction, production `StudentTNLL`, optimization, validation-only selection, and evaluation; Full/LR/Block cross-family learners consume exactly the same teacher artifact, while Graph runs matched only.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyTorch, e3nn through existing compatibility wrappers, pytest, ruff, and existing equivcompiler planning/distribution/evaluation modules.

## Global Constraints

- Teacher defines the data-generating distribution independently; the compiler is used only on the learner side.
- `experiments/independent_teacher_oracle.py` may import only standard-library modules, NumPy, and SciPy.
- Oracle arrays use FP64; learner training uses existing FP32 operator/NLL algebra and formal operator evaluation uses FP64.
- Reuse existing `StudentTNLL`, compiler family assembly, and metrics; no second production probability implementation.
- Formal families are Full, rank-2-plus-isotropic, isotypic-block, and three-node chain Graph precision with fixed `nu=5`.
- Full/LR/Block cross-family rows for a teacher seed share one exact `.npz` hash.
- Formal seeds are `0,1,2`; test data never control checkpoint selection.
- Gates in the approved design are immutable after learner results are observed.
- Do not modify or stage existing plotting/figure changes.
- Do not start dielectric, ITOP, elasticity, certificate-checker, or performance work.

---

### Task 1: Pure NumPy/SciPy coordinate and family oracle

**Files:**
- Create: `experiments/independent_teacher_oracle.py`
- Create: `tests/test_independent_teacher_oracle.py`

**Interfaces:**
- Produces: `OracleProtocol`, `OracleDataset`, and `build_oracle_dataset(family: str, seed: int, protocol: OracleProtocol) -> OracleDataset`.
- Produces family constructors returning independent FP64 mean, scatter, and optional precision arrays.
- Consumes only NumPy, SciPy, and the standard library.

- [ ] **Step 1: Write the failing import-isolation test**

```python
def test_oracle_source_has_no_repository_or_torch_imports():
    tree = ast.parse(Path("experiments/independent_teacher_oracle.py").read_text())
    forbidden = {"torch", "e3nn", "equivcompiler", "spd_maps", "distributions", "evaluation"}
    roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert roots.isdisjoint(forbidden)
```

- [ ] **Step 2: Run it and confirm failure**

Run: `pytest tests/test_independent_teacher_oracle.py::test_oracle_source_has_no_repository_or_torch_imports -v`

Expected: FAIL because the oracle module is absent.

- [ ] **Step 3: Add protocol and dataset dataclasses**

```python
@dataclass(frozen=True)
class OracleProtocol:
    train_contexts: int = 128
    train_replicates: int = 32
    validation_contexts: int = 64
    validation_replicates: int = 64
    test_contexts: int = 128
    test_replicates: int = 128
    calibration_draws: int = 65_536
    calibration_trials: int = 2_048
    nu: float = 5.0

@dataclass(frozen=True)
class OracleDataset:
    family: str
    seed: int
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]
```

Use `np.random.SeedSequence(seed).spawn(...)` for disjoint deterministic split streams. Validate positive sizes, supported families, and `nu > 2`.

- [ ] **Step 4: Add the explicit scalar-plus-STF coordinate test**

```python
def test_rank2_basis_is_orthonormal_and_matches_public_coordinate_contract():
    basis = oracle.rank2_cartesian_basis()
    np.testing.assert_allclose(np.einsum("aij,bij->ab", basis, basis), np.eye(6), atol=1e-14)
    np.testing.assert_allclose(basis, _repository_rank2_basis_as_numpy(), atol=1e-12)
```

The oracle basis uses explicit square-root constants; only this test helper may import the repository/e3nn convention.

- [ ] **Step 5: Implement direct family equations**

Implement `full_scatter`, `low_rank_scatter`, `isotypic_block_scatter`, and `graph_precision_scatter`. Use `scipy.linalg.expm` for Full and Graph, direct `sigma2 I + L L^T` for LR, `diag(k0,k2 I5)` for Block, and direct incidence assembly plus Cholesky solve for Graph. Teacher mean is an explicit FP64 zero array.

- [ ] **Step 6: Add SPD, family-membership, and equivariance tests**

```python
@pytest.mark.parametrize("family", ["full", "low_rank", "isotypic_block", "graph_precision"])
def test_oracle_family_is_spd_and_equivariant(family):
    dataset = build_oracle_dataset(family, 3, SMOKE_PROTOCOL)
    assert np.linalg.eigvalsh(dataset.arrays["test_scatter"]).min() > 1e-10
    assert dataset.metadata["self_checks"]["equivariance_relative_max"] <= 5e-10
```

Also test LR residual eigenvalue multiplicity, Block structure, and Graph precision equality to the incidence formula.

- [ ] **Step 7: Add independent Student-t sampling and radial tests**

Sampling uses NumPy normal/chi-square draws and SciPy Cholesky factors. Test teacher Coverage90/95 against broad smoke tolerances without calling repository distributions.

- [ ] **Step 8: Run tests and commit**

Run:
```bash
pytest tests/test_independent_teacher_oracle.py -v
git add experiments/independent_teacher_oracle.py tests/test_independent_teacher_oracle.py
git commit -m "add independent NumPy covariance teacher"
```

Expected: tests PASS; commit contains only these files.

---

### Task 2: Immutable artifacts and pre-learner gates

**Files:**
- Modify: `experiments/independent_teacher_oracle.py`
- Modify: `tests/test_independent_teacher_oracle.py`

**Interfaces:**
- Produces: `write_oracle_artifact(dataset, output_dir, source) -> dict[str, object]`.
- Produces: `load_oracle_artifact(npz_path, manifest_path) -> OracleDataset`.
- Manifest records NPZ hash, split hashes, oracle version, equations/coefficients, source provenance, self-checks, and frozen coverage tolerances.

- [ ] **Step 1: Write failing reproducibility and corruption tests**

Generate the same artifact twice and require identical NPZ hashes. Corrupt a copied NPZ byte and require `ValueError("oracle artifact hash mismatch")`.

- [ ] **Step 2: Run the tests and confirm failure**

Run: `pytest tests/test_independent_teacher_oracle.py -k artifact -v`

Expected: FAIL because artifact functions are absent.

- [ ] **Step 3: Implement canonical serialization**

Save sorted FP64 arrays with `allow_pickle=False`, hash final NPZ bytes, atomically replace the JSON manifest, and record the manifest's own hash later in learner results to avoid self-reference.

- [ ] **Step 4: Freeze teacher-only paired Monte Carlo tolerances**

For Coverage90/95, use `scipy.stats.f.ppf` under `q/d ~ F(d,nu)`. Simulate paired exact-model coverage counts at the formal held-out sample count and save the empirical 99th percentile of their absolute difference before learner construction.

- [ ] **Step 5: Test split independence and manifest completeness**

Require disjoint split IDs, stable split hashes, source commit/dirty state, family coefficients, finite/SPD checks, and frozen tolerances with no learner fields.

- [ ] **Step 6: Run tests and commit**

Run:
```bash
pytest tests/test_independent_teacher_oracle.py -v
git add experiments/independent_teacher_oracle.py tests/test_independent_teacher_oracle.py
git commit -m "record immutable independent teacher artifacts"
```

---

### Task 3: Compiler learner adapter and validation-only selection

**Files:**
- Modify: `experiments/synthetic_covariance_benchmark.py`
- Modify: `tests/test_synthetic_covariance_benchmark.py`

**Interfaces:**
- Consumes verified oracle NPZ/manifest artifacts.
- Produces `run_independent_pair(teacher_artifact, learner_name: str, *, steps: int, patience: int, device: str) -> dict[str, object]`.
- Produces `run_independent_teacher_matrix(artifact_dir, families, seeds, ...) -> dict[str, object]`.
- Retains legacy `run_case`/`run_pair`, but marks them as legacy and excludes them from formal evidence.

- [ ] **Step 1: Write the failing shared-dataset test**

```python
def test_cross_family_learners_share_exact_teacher_artifact(tmp_path):
    artifact = make_smoke_oracle_artifact(tmp_path, family="full", seed=7)
    rows = [run_independent_pair(artifact, family, steps=1, patience=1, device="cpu")
            for family in ("full", "low_rank", "isotypic_block")]
    assert {row["oracle_npz_sha256"] for row in rows} == {artifact["npz_sha256"]}
```

Also require explicit representation-contract errors for Graph/non-Graph mismatches.

- [ ] **Step 2: Run and confirm failure**

Run: `pytest tests/test_synthetic_covariance_benchmark.py -k independent -v`

Expected: FAIL because the independent learner interface is absent.

- [ ] **Step 3: Extract only learner-side reusable training**

Reuse compiler plan/head construction, existing `StudentTNLL`, optimizer, gradient clipping, and metrics. Formal data must never pass through compiler-teacher construction or `_prepare_head(teacher, teacher)`.

- [ ] **Step 4: Implement validation-only selection**

Evaluate validation NLL each epoch, copy the best state to CPU, stop after `patience`, then restore the selected state before loading test arrays. Record selected epoch, best validation NLL, last epoch, and `selection_split="validation"`.

- [ ] **Step 5: Materialize formal FP64 metrics and gates**

Report mean/p90 relative operator error, Coverage90/95, NLL, equivariance, orthogonal scatter/NLL invariance, finite checks, and hashes. Graph uses precision as primary and scatter as secondary. Matched rows receive numeric/recovery/coverage/provenance/overall booleans; cross-family rows set `overall=None`.

- [ ] **Step 6: Run regression tests and commit**

Run:
```bash
pytest tests/test_synthetic_covariance_benchmark.py -v
git add experiments/synthetic_covariance_benchmark.py tests/test_synthetic_covariance_benchmark.py
git commit -m "evaluate compiler learners on independent teachers"
```

---

### Task 4: Formal CLI and test-only NLL cross-check

**Files:**
- Modify: `experiments/synthetic_covariance_benchmark.py`
- Modify: `tests/test_independent_teacher_oracle.py`
- Modify: `tests/test_synthetic_covariance_benchmark.py`

**Interfaces:**
- Formal CLI uses `--teacher-backend independent_numpy --seeds 0,1,2`, generates/reuses hash-verified oracle artifacts, and atomically writes one result JSON.

- [ ] **Step 1: Add a failing test-only SciPy NLL cross-check**

For a fixed FP64 Student-t example, compute a reference with `scipy.special.gammaln`, `np.linalg.slogdet`, and `np.linalg.solve`; compare to existing `StudentTNLL` at `1e-10`. Keep the reference inside the test file.

- [ ] **Step 2: Add failing CLI contract tests**

Require `kind="independent_numpy_teacher_scatter_recovery"`, `teacher_side="numpy_scipy_only"`, `learner_side="public_compiler"`, and 12 required matched rows.

- [ ] **Step 3: Implement formal CLI and atomic output**

Add oracle directory, protocol sizes, patience, reuse validation, and independent backend arguments. Refuse reused artifacts on schema/hash/family/seed/nu/split mismatch. Write via sibling temporary JSON and `Path.replace`.

- [ ] **Step 4: Verify row accounting**

Formal output contains 12 matched rows (3 seeds × 4 families) and 18 mismatched diagnostics (3 seeds × 3 rank-2 teachers × 2 mismatched learners), without duplicating matched rows.

- [ ] **Step 5: Run targeted verification and commit**

Run:
```bash
pytest tests/test_independent_teacher_oracle.py tests/test_synthetic_covariance_benchmark.py -v
pytest tests/test_public_compiler_api.py tests/test_graph_precision.py -v
ruff check experiments/independent_teacher_oracle.py experiments/synthetic_covariance_benchmark.py tests/test_independent_teacher_oracle.py tests/test_synthetic_covariance_benchmark.py
git add experiments/synthetic_covariance_benchmark.py tests/test_independent_teacher_oracle.py tests/test_synthetic_covariance_benchmark.py
git commit -m "add formal independent recovery protocol"
```

Expected: all tests/lint PASS.

---

### Task 5: Smoke, server formal run, and evidence gate

**Files:**
- Create after formal results: `docs/independent_teacher_recovery_results_20260811.md`
- Do not modify manuscript or plotting files.

**Interfaces:**
- Produces server oracle artifacts, learner outputs/logs/result JSON, downloaded local results, and a gate document.

- [ ] **Step 1: Run a local CPU smoke**

Use explicit small sizes and one seed under `results/independent_teacher_recovery_smoke_20260811`. Mark it development-only. Do not use smoke values as evidence or alter formal thresholds.

- [ ] **Step 2: Verify before push**

Run:
```bash
pytest tests/test_independent_teacher_oracle.py tests/test_synthetic_covariance_benchmark.py tests/test_public_compiler_api.py tests/test_graph_precision.py -v
ruff check experiments/independent_teacher_oracle.py experiments/synthetic_covariance_benchmark.py tests/test_independent_teacher_oracle.py tests/test_synthetic_covariance_benchmark.py
git diff --check
```

- [ ] **Step 3: Commit any minimal smoke fix and push**

Stage only files in this plan. Record the exact source commit. Never stage plotting changes or `plotting/style.py.bak`.

- [ ] **Step 4: Create a clean server worktree**

Use `ssh dbcloud`; fetch without resetting the dirty main checkout; create detached `/home/workspace/lrh/Tpami_independent_teacher_<commit7>` at the exact pushed commit.

- [ ] **Step 5: Run formal three-seed recovery**

Activate `/home/workspace/lrh/miniconda3/envs/equivcompiler`, export `EQUIVCOMPILER_DATA_ROOT=/home/workspace/lrh/DATA/Tpami`, and write below `/home/workspace/lrh/RESULTS/Tpami/Synthetic/independent_teacher_<commit7>`. Use one available GPU and formal defaults. Do not launch dielectric.

- [ ] **Step 6: Verify and download artifacts**

Require clean exact source, 12 matched rows, 18 diagnostics, shared hashes, finite outputs, validated hashes, validation-only selection, FP64 evaluation, and all gate fields. Download to `E:\\CODE\\Tpami\\results\\independent_teacher_recovery_<commit7>` and independently re-hash every artifact.

- [ ] **Step 7: Write the evidence document**

Separate existing evidence, new evidence, supported inference, rejected explanation, unresolved items, and formal gate. If any matched row fails, stop before dielectric. If all pass, state that the dielectric factorial is permitted but not executed.

- [ ] **Step 8: Verify, commit, and push evidence**

Run targeted tests and `git diff --check`; stage only the evidence document and any narrowly required summary JSON. Do not update the manuscript in this plan.


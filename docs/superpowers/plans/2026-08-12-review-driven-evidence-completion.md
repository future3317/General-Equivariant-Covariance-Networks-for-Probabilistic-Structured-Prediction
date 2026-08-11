# Review-Driven Evidence Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce artifact-backed ITOP paired statistics, fixed-nu dielectric sensitivity evidence, and a tightly scoped TPAMI update without starting high-cost new model training.

**Architecture:** Reuse saved prediction artifacts and existing evaluation primitives. A small audit module will load and validate prediction cells, compute per-example metric arrays, and bootstrap paired contrasts. A separate dielectric adapter will consume saved mean/scale/target tensors and call the existing Student-t likelihood/diagnostic functions for a fixed nu grid. Evidence JSON/Markdown is generated from these outputs; the manuscript is then edited only where the new artifacts support a claim.

**Tech Stack:** Python 3.11, PyTorch, NumPy, SciPy, existing `evaluation` package, pytest, LaTeX/IEEEtran, Poppler rendering.

## Global Constraints

- Do not modify or stage existing dirty files in `E:\CODE\Tpami` except newly added audit code/tests/docs and the explicitly generated result report.
- Do not start new ITOP/elasticity/dielectric model training, E3b, mixture expansion, second-group experiments, or external UQ baselines.
- Reuse existing Student-t NLL, Energy Score, calibration, and risk-coverage implementations.
- ITOP six-head rows are single-seed factorial evidence; Full/LR/Graph three-seed rows are paired robustness evidence and must not be merged into a false six-head three-seed table.
- Fixed-nu dielectric scans are post-hoc sensitivity diagnostics, not validation-selected training.
- Test selection is prohibited; validation metadata may be reported only when already present in the artifact.

---

### Task 1: Add failing tests for paired ITOP artifact audit

**Files:**
- Create: `tests/test_review_evidence_audit.py`
- Test target to add later: `scripts/review_evidence_audit.py`

**Interfaces:**
- The module will expose `bootstrap_mean_interval(values, *, seed, samples, confidence)`.
- The module will expose `paired_difference(left, right)`.
- The module will expose `validate_prediction_pair(left, right)`.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest

from scripts.review_evidence_audit import (
    bootstrap_mean_interval,
    paired_difference,
    validate_prediction_pair,
)


def test_paired_difference_preserves_frame_alignment():
    left = {"frame_index": np.array([3, 1]), "target": np.zeros((2, 4))}
    right = {"frame_index": np.array([3, 1]), "target": np.ones((2, 4))}
    diff = paired_difference(left, right)
    assert diff.shape == (2,)
    np.testing.assert_allclose(diff, [0.0, 0.0])


def test_validate_prediction_pair_rejects_mismatched_targets():
    left = {"frame_index": np.array([0]), "target": np.zeros((1, 4))}
    right = {"frame_index": np.array([0]), "target": np.ones((1, 4))}
    with pytest.raises(ValueError, match="target"):
        validate_prediction_pair(left, right)


def test_bootstrap_mean_interval_is_deterministic_and_contains_mean():
    values = np.array([-2.0, 0.0, 2.0])
    interval = bootstrap_mean_interval(values, seed=7, samples=2000, confidence=0.95)
    assert interval[0] <= values.mean() <= interval[1]
    assert interval == bootstrap_mean_interval(values, seed=7, samples=2000, confidence=0.95)
```

- [ ] **Step 2: Run the focused tests and verify the expected missing-module failure**

Run: `python -m pytest tests/test_review_evidence_audit.py -q`

Expected: collection fails because `scripts.review_evidence_audit` does not yet exist.

---

### Task 2: Implement ITOP six-head summary and paired bootstrap

**Files:**
- Create: `scripts/review_evidence_audit.py`
- Modify: `tests/test_review_evidence_audit.py` only if a test exposes an interface mismatch

**Interfaces:**
- `paired_difference(left, right) -> np.ndarray` validates identical frame IDs and targets, then returns per-frame scalar error differences.
- `bootstrap_mean_interval(values, *, seed: int, samples: int, confidence: float) -> tuple[float, float]` uses paired resampling of the supplied difference vector.
- CLI arguments: `--itop-factorial-root`, `--itop-robustness-root`, `--output`.
- Output JSON contains `single_seed_six_head`, `three_seed_paired_bootstrap`, `provenance`, and explicit evidence-status sections.

- [ ] **Step 1: Implement minimal helpers**

Use `torch.load(..., map_location="cpu", weights_only=True)` for prediction artifacts, convert only needed tensors to NumPy, and use existing metric functions for exact NLL/energy/risk metrics. Do not implement a new Student-t density.

- [ ] **Step 2: Run focused tests and verify green**

Run: `python -m pytest tests/test_review_evidence_audit.py -q`

Expected: all focused tests pass.

- [ ] **Step 3: Run the artifact audit against the existing ITOP roots**

Run: `python scripts/review_evidence_audit.py --itop-factorial-root E:\CODE\Tpami\results\itop_reviewer_factorial_3844f99 --itop-robustness-root E:\CODE\Tpami\results\itop_reviewer_final_full_f9e02f3 --output E:\CODE\Tpami\results\itop_review_evidence_20260812`

Expected: six-head single-seed rows are emitted only for complete cells, Full/LR/Graph paired intervals are emitted for seeds 42/43/44, and the report records that the studies are separate.

---

### Task 3: Add failing tests and fixed-nu dielectric sensitivity adapter

**Files:**
- Modify: `tests/test_review_evidence_audit.py`
- Create: `scripts/audit_dielectric_fixed_nu_sensitivity.py`

**Interfaces:**
- `student_t_nll_from_prediction(prediction, nu) -> float` calls the existing Student-t distribution/NLL implementation through the established evaluation path.
- CLI arguments: `--prediction-root`, `--output`, `--nu 3 5 10 30`.

- [ ] **Step 1: Add a failing test for nu-grid validation**

```python
def test_fixed_nu_grid_rejects_nonpositive_dof():
    from scripts.audit_dielectric_fixed_nu_sensitivity import validate_nu_grid
    with pytest.raises(ValueError, match="nu"):
        validate_nu_grid([0.0, 5.0])
```

- [ ] **Step 2: Run the focused test and verify it fails because the adapter is absent**

Run: `python -m pytest tests/test_review_evidence_audit.py::test_fixed_nu_grid_rejects_nonpositive_dof -q`

Expected: import/collection failure for the missing adapter.

- [ ] **Step 3: Implement the minimal adapter**

Load saved Full Student-t prediction tensors and evaluate the fixed grid using existing `distributions.student_t` or `evaluation` interfaces. Record source prediction hash, source checkpoint hash, `selection: none`, and whether the prediction artifact contains validation data. Do not fit nu on test.

- [ ] **Step 4: Run tests and execute the post-hoc scan**

Run focused pytest, then execute the CLI for each complete Full-t seed available in the factorial artifact. Compare the output against the recorded `nu=5` NLL within the artifact tolerance.

Expected: finite output for all requested nu values and exact agreement at nu=5 with the recorded Student-t semantics.

---

### Task 4: Generate evidence reports and reproducibility table

**Files:**
- Create: `docs/review_evidence_completion_20260812.md`
- Create: `results/itop_review_evidence_20260812/*` via the audit CLI
- Create: `results/dielectric_fixed_nu_sensitivity_20260812/*` via the sensitivity CLI

**Interfaces:**
- Reports must separate `existing evidence`, `new evidence`, `supported inference`, `rejected explanation`, and `unresolved`.

- [ ] **Step 1: Review generated JSON for artifact hashes and sample counts**
- [ ] **Step 2: Write compact Markdown evidence report without internal debug/process language**
- [ ] **Step 3: Add a compact reproducibility table source fragment for the manuscript**

---

### Task 5: Update TPAMI manuscript conservatively

**Files:**
- Modify: `E:\PAPER\General Equivariant Covariance Networks for Probabilistic Structured Prediction\bare_jrnl_new_sample4.tex`
- Modify only if needed: manuscript table/figure assets already inside that repository

**Interfaces:**
- Add the complete single-seed ITOP factorial table only if it fits at readable font size; otherwise place it in the appendix and keep a compact main-text contrast.
- Add paired bootstrap intervals for Graph-t vs Full-t and LR-t vs Graph-t using artifact-backed numbers.
- State the dielectric fixed-nu scan as sensitivity, and state the validation-fitted temperature result as a negative held-out control.
- Remove or rewrite unsupported “conditional-nu diagnostics” wording if the formal result is not cited.
- Change low-rank limitation wording so strict statistical subfamily is not called approximate; reserve approximate for truncated lowering/fidelity.
- Correct `q_full/q_act=1/0` to `q_full=1, q_act=0` and tighten claim scope to orthogonal-basis/O(3)-validated backend.
- Do not add external baseline, five-seed, second-group, or unverified convergence claims.

- [ ] **Step 1: Patch only evidence-backed prose/tables**
- [ ] **Step 2: Compile the manuscript with the existing IEEEtran workflow**
- [ ] **Step 3: Render the complete PDF and inspect all changed pages and wide tables**

---

### Task 6: Verify, preserve unrelated work, and commit minimal changes

**Files:**
- Verify: code repo dirty-state boundaries and manuscript repo diff

- [ ] **Step 1: Run focused and relevant full tests**

Run: `python -m pytest tests/test_review_evidence_audit.py tests/test_evaluation.py tests/test_itop_training_control.py -q`

- [ ] **Step 2: Run `ruff check` on new/modified code and `git diff --check`**
- [ ] **Step 3: Run the manuscript LaTeX build and check for undefined references/citations**
- [ ] **Step 4: Inspect rendered PDF pages for table readability, clipping, and float placement**
- [ ] **Step 5: Commit only new audit/report files in the code repo and intended manuscript files in the paper repo**


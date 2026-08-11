# Statistical Misspecification Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in global/conditional Student-t and exact K=2 mixture repairs to the existing frozen uncertainty protocols, then run one-seed dielectric/ITOP falsification with existing spectral controls and optional observation descriptors.

**Architecture:** Keep compiler IR, typed families, SPD maps, and exact lowerings unchanged. Build distribution-law modules that consume compiled SPD sufficient statistics, and frozen uncertainty heads that emit invariant degrees of freedom, invariant mixture weights, and component parameter projections. Extend the existing E1 runner rather than creating a second training pipeline.

**Tech Stack:** Python, PyTorch, e3nn-compatible O(3) linear layers, existing `SPDMap`, pytest, existing E1 cache/evaluation utilities.

## Global Constraints

- The legacy `fixed` variant and default CLI behavior remain unchanged.
- No compiler rewrite, new default family, moment matching, averaged component NLL, or GT/visibility descriptor.
- Every new path is selected by an explicit variant/config flag and records its semantics.
- All components use the existing certified SPD map and its exact lowering.
- Student-t diagnostics use `nu > 2`; training/evaluation retains normalized density constants.
- Existing dirty files in the checkout are preserved and never staged.

---

### Task 1: Add law-level tests before implementation

**Files:**
- Create: `tests/test_statistical_misspecification_repair.py`
- Test: `distributions/mixture.py`, `models/frozen_distribution_readout.py`, `data/frozen_distribution_features.py`

- [ ] **Step 1: Write tests for exact two-component logsumexp, global/conditional ν, O(3) invariance, centered component means, and descriptor rejection.**
- [ ] **Step 2: Run `pytest tests/test_statistical_misspecification_repair.py -q` and record the expected import/attribute failures.**

### Task 2: Implement normalized finite-mixture Student-t primitive

**Files:**
- Modify: `distributions/student_t.py`
- Create: `distributions/mixture.py`
- Modify: `distributions/__init__.py`
- Test: `tests/test_statistical_misspecification_repair.py`

- [ ] **Step 1: Implement `FiniteMixtureStudentTNLL.forward(component_means, component_params, target, spd_map, weights, nu)` using flattened `spd_map.statistics`, `student_t_log_prob_from_statistics`, and exact `torch.logsumexp`.**
- [ ] **Step 2: Return detached component log-probability, mixture log-probability, responsibilities, logdet, quadratic, weights, and nu diagnostics without changing `StudentTNLL`.**
- [ ] **Step 3: Run the law tests and verify the test with a hand-computed logsumexp now passes.**

### Task 3: Add feature-gated frozen repair heads

**Files:**
- Modify: `models/frozen_distribution_readout.py`
- Modify: `models/__init__.py` only if public exports are needed
- Test: `tests/test_statistical_misspecification_repair.py`

- [ ] **Step 1: Add a trainable global-ν readout and invariant mixture-logit readout initialized to the legacy fixed law/equal weights.**
- [ ] **Step 2: Add shared-mean K=2 and centered multimodal K=2 heads using two existing O(3) linear parameter projections and the exact mixture primitive.**
- [ ] **Step 3: Add optional invariant descriptor concatenation only at the uncertainty readout input; reject descriptor fields named `visible`, `target`, or `label`.**
- [ ] **Step 4: Run rotation/reflection, finite-SPD, gradient, ν>2, and weighted-mean-preservation tests.**

### Task 4: Extend the existing E1 runner without changing the fixed path

**Files:**
- Modify: `scripts/run_frozen_distribution_e1.py`
- Modify: `evaluation/ensemble.py`
- Modify: `data/frozen_distribution_features.py`
- Test: `tests/test_run_frozen_distribution_e1.py`

- [ ] **Step 1: Add explicit `global_nu`, `shared_mean_mixture`, and `multimodal_mean_mixture` variants while retaining `fixed`, `conditional_nu`, and existing spectral variants.**
- [ ] **Step 2: Serialize scalar/sample/component ν, component scales, weights, and exact-law metadata in prediction artifacts.**
- [ ] **Step 3: Extend mixture sampling to sample categorical components with invariant sample weights and component-valued ν; keep old `sample_ensemble` calls compatible.**
- [ ] **Step 4: Add optional descriptor payload loading by sample-id with no default cache mutation.**
- [ ] **Step 5: Run runner unit tests and a CPU smoke run on the existing synthetic fixture.**

### Task 5: Add descriptor-cache preparation using existing geometry code

**Files:**
- Create: `scripts/prepare_itop_observation_descriptors.py`
- Modify: `data/observation_descriptors.py` only for reusable validation helpers
- Test: `tests/test_itop_mixture_mechanism.py` or the new repair test file

- [ ] **Step 1: Build `train/val/test/ood` descriptor tensors from existing point/depth caches and write hashes/sample IDs.**
- [ ] **Step 2: Exclude visibility/label fields from the serialized model input and record the exact descriptor names.**
- [ ] **Step 3: Run descriptor alignment and provenance tests.**

### Task 6: Run one-seed falsification and write comparison report

**Files:**
- Create: `scripts/run_statistical_misspecification_repair.py`
- Create: `docs/statistical_misspecification_repair_report_20260812.md`
- Create: results under `results/statistical_misspecification_repair_20260812/`

- [ ] **Step 1: Run dielectric fixed/global/conditional/shared-mixture using the same frozen cache, IDs, validation-only selection, and one seed.**
- [ ] **Step 2: Run ITOP Full-t/Graph-t/Graph conditional-ν/Graph K=2/Graph K=2 plus descriptors on the declared small Side subset; Top is evaluation-only.**
- [ ] **Step 3: Run existing centered spectral controls only if the runner and artifacts pass the law gates.**
- [ ] **Step 4: Compute normalized NLL, Energy, coverage, MACE, law-correct q/radial PIT, whitened defect, direction test, component alignment, and ITOP shift metrics.**
- [ ] **Step 5: Stop each branch on a predeclared negative result and write answers to the four requested questions without upgrading to three seeds/full data.**

### Task 7: Verify and integrate only owned changes

**Files:**
- Modify only the files listed above; do not stage existing dirty files.

- [ ] **Step 1: Run focused tests, full relevant pytest, Ruff, and `git diff --check`.**
- [ ] **Step 2: Audit compiler regression tests and legacy fixed-variant reproduction.**
- [ ] **Step 3: Commit only owned code/tests/docs/results with a focused message; leave unrelated dirty files untouched.**

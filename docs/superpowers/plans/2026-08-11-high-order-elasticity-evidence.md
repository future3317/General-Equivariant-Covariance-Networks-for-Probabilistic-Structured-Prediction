# High-Order Elasticity Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce audited end-to-end rank-4 elasticity training evidence and low-cost paired ITOP family statistics without repeating completed uncertainty experiments.

**Architecture:** Extend the existing elasticity trainer into a three-arm typed study with a shared deterministic data contract, reusable evaluator, and staged orchestration. Reuse production compiler/distribution/evaluation primitives; add only experiment control, serialization, and paired artifact analysis.

**Tech Stack:** Python 3.11, PyTorch, PyG, e3nn compatibility layer, NumPy/SciPy, pytest.

## Global Constraints

- Preserve the current dirty primary checkout; work only in this isolated worktree.
- Reuse `plan_readout`, `StudentTNLL`, SPD maps, Energy Score, coverage, elliptical diagnostics, and provenance helpers.
- Keep operator assembly and NLL in FP32; use FP64 only for audit reconstruction.
- Side/Top ITOP analysis is artifact-only; no new ITOP or dielectric training.
- Pilot before formal training; do not expand after a failed gate.

---

### Task 1: Deterministic elasticity data and model contract

**Files:**
- Modify: `data/elasticity_dataset.py`
- Modify: `scripts/train_elasticity.py`
- Create: `tests/test_elasticity_training_control.py`

**Interfaces:**
- Produces: deterministic seeded subset indices, `build_elasticity_model(args)`, and a common forward/loss result schema for deterministic and probabilistic arms.

- [ ] Write failing tests proving identical seed/split inputs across arms, deterministic mean-only construction, and compiled Full/Low-rank Student-t schemas.
- [ ] Run `python -m pytest tests/test_elasticity_training_control.py -q` and confirm failures are caused by missing interfaces.
- [ ] Implement seeded subset selection and the minimal common model builder using existing `DeterministicHead` and `plan_readout`.
- [ ] Re-run the focused tests and the existing compiler/head tests.
- [ ] Commit the data/model contract.

### Task 2: Audited prediction evaluation

**Files:**
- Create: `scripts/evaluate_elasticity.py`
- Create: `tests/test_elasticity_evaluation.py`

**Interfaces:**
- Consumes: prediction dictionaries with `sample_id`, `mean`, `target`, and optional `params` plus the bound compiled model.
- Produces: `evaluate_elasticity_predictions(...) -> dict` with point, proper-score, coverage, calibration, elliptical, finite, and resource fields.

- [ ] Write failing synthetic tests for deterministic point metrics and Student-t NLL/Energy/Coverage/diagnostic output.
- [ ] Verify the tests fail because the evaluator is absent.
- [ ] Implement chunked prediction collection and metrics by composing existing evaluation functions.
- [ ] Add FP64 scatter finite/SPD checks and prediction SHA recording without storing duplicate dense scatters.
- [ ] Run focused tests, then `tests/test_evaluation.py`, `tests/test_elliptical_diagnostics.py`, and distribution tests.
- [ ] Commit the evaluator.

### Task 3: Training artifacts and staged runner

**Files:**
- Modify: `scripts/train_elasticity.py`
- Create: `scripts/run_elasticity_study.py`
- Modify: `tests/test_elasticity_training_control.py`

**Interfaces:**
- Produces: arm artifacts (`args/environment/schema/history/metrics/predictions/checkpoint/log`) and a study manifest for `deterministic`, `low_rank_student_t`, and `full_student_t`.

- [ ] Write failing tests for CLI arm mapping, validation-only selection, required artifact checks, and Gate-1 decision logic.
- [ ] Verify expected failures.
- [ ] Add seed control, runtime/memory measurement, atomic JSON writes, source/data/split/checkpoint/prediction provenance, and evaluator invocation to the trainer.
- [ ] Implement the sequential staged runner with no automatic formal expansion unless the pilot gate passes.
- [ ] Run focused tests and a CPU one-batch smoke for all three arms.
- [ ] Commit orchestration and artifacts.

### Task 4: ITOP paired family analysis

**Files:**
- Create: `scripts/analyze_itop_family_tradeoff.py`
- Create: `tests/test_itop_family_tradeoff.py`

**Interfaces:**
- Produces: paired bootstrap CI and Pareto records from audited Full-t/Graph-t prediction and runtime artifacts.

- [ ] Write failing tests for target/order mismatch rejection, paired bootstrap reproducibility, and Pareto dominance labeling.
- [ ] Verify expected failures.
- [ ] Implement artifact loading by reusing existing ITOP prediction audit helpers and exact per-sample Student-t semantics.
- [ ] Run focused tests and execute against the local mirrored artifacts.
- [ ] Commit the artifact-only analysis.

### Task 5: Pilot performance gate

**Files:**
- Create: `docs/elasticity_training_performance_report_20260811.md`

**Interfaces:**
- Consumes: three-arm pilot artifacts and GPU telemetry.
- Produces: accepted/rejected/inconclusive bottleneck decision and exact formal-run command.

- [ ] Capture server environment, free GPU, data counts, and a one-batch eager baseline.
- [ ] Run the seed-42 1,024/256 pilot for all three arms.
- [ ] Profile only if throughput or memory blocks the gate; patch one measured cause and rerun the identical benchmark.
- [ ] Verify all Gate-1 scientific and artifact conditions.
- [ ] Commit the performance report and pilot evidence summary.

### Task 6: Formal confirmation and evidence report

**Files:**
- Create: `docs/elasticity_end_to_end_results_20260811.md`
- Create: `docs/itop_paired_family_tradeoff_20260811.md`

**Interfaces:**
- Consumes: formal three-seed elasticity artifacts and ITOP artifact-only analysis.
- Produces: evidence matrix split into existing evidence, new evidence, supported inference, rejected explanation, and unresolved.

- [ ] Run the formal study only if Gate 1 passed.
- [ ] Audit all nine arm-seed cells, hashes, schemas, finite values, selection, and metrics.
- [ ] Download compact artifacts to `results/elasticity_end_to_end_<commit>`; retain full checkpoints on the server if large.
- [ ] Run the ITOP paired analysis and record bootstrap/Pareto results.
- [ ] Run full relevant tests, Ruff on changed files, and `git diff --check`.
- [ ] Commit and push the minimal code/evidence changes; manuscript updates remain a separate gate.

# Dielectric Frozen-Family Factorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an audited isotropic/block/rank-2/full by Gaussian/Student-t dielectric factorial while freezing one immutable feature representation and mean.

**Architecture:** Generalize the existing frozen-mean scatter readout to compose an already compiled SPD map with either existing elliptical objective, then extract the ordinary single-ellipse training/evaluation path from the E1 runner into one reusable task-neutral engine. A thin factorial CLI supplies compiler family/law schemas, and a separate aggregator verifies common artifacts and computes predeclared comparisons. Execution is staged as CPU smoke, one-seed full-split pilot, and only then a fresh three-seed formal run.

**Tech Stack:** Python 3.11, PyTorch, e3nn-compatible typed projections, existing `equivcompiler` policies, existing Gaussian/Student-t distributions and evaluation package, pytest, ruff.

## Global Constraints

- Freeze the exact cached `H`, `mu`, train/validation/test tensors, and sample IDs from `dielectric_unified_full_cache`.
- Families are isotropic=`LowRankCovariance(0)`, block=`IsotypicBlockCovariance()`, low-rank=`LowRankCovariance(2)`, and full=`CenteredSpectralWindowCovariance(-2,2,-8,8)`.
- Laws are existing Gaussian and fixed-`nu=5` Student-t; do not implement another density or SPD map.
- Formal seeds are `42,43,44`; all arms use batch size 128, AdamW `lr=5e-4`, weight decay `1e-5`, scheduler factor 0.5/patience 2, and early-stopping patience 5.
- Validation NLL is the only model-selection quantity. Test results never change training or arm selection.
- Stage 1 uses the complete splits, seed 42, at most 20 epochs, and must pass the operational gate before Stage 2.
- Stage 2 reruns all 24 arms fresh with at most 60 epochs; Stage-1 artifacts are development-only.
- Do not update the manuscript, add mixtures/conditional `nu`/calibration, or retrain a backbone.

---

### Task 1: Generic frozen-mean elliptical scatter readout

**Files:**
- Modify: `models/frozen_distribution_readout.py`
- Modify: `tests/test_frozen_distribution_readout.py`

**Interfaces:**
- Consumes: existing `GaussianNLL`, `StudentTNLL`, `SPDMap`, feature irreps, parameter irreps, frozen means, and targets.
- Produces: `FrozenMeanScatterElliptical(feature_irreps, parameter_irreps, spd_map, *, distribution, student_t_dof=5.0)` and keeps `FrozenMeanScatterStudentT` as a compatibility wrapper.

- [ ] **Step 1: Write failing objective-composition tests**

Add tests which initialize one legal `0e+2e` feature batch, fixed means/targets,
and a compiler-built SPD map, then assert:

```python
gaussian = FrozenMeanScatterElliptical(
    feature_irreps,
    parameter_irreps,
    spd_map,
    distribution="gaussian",
)
student = FrozenMeanScatterElliptical(
    feature_irreps,
    parameter_irreps,
    spd_map,
    distribution="student_t",
    student_t_dof=5.0,
)
assert isinstance(gaussian.objective, GaussianNLL)
assert isinstance(student.objective, StudentTNLL)
assert torch.isfinite(gaussian(features, mean, target)["loss"])
assert torch.isfinite(student(features, mean, target)["loss"])
```

Also compare each returned loss to a direct call to the corresponding existing
objective on the exact projected parameters. Assert that an unsupported law
and Student-t `nu <= 2` are rejected.

- [ ] **Step 2: Run the focused tests and confirm the missing-class failure**

Run: `python -m pytest tests/test_frozen_distribution_readout.py -q`

Expected: failure because `FrozenMeanScatterElliptical` is not defined.

- [ ] **Step 3: Implement the minimal composition**

Add one class whose constructor selects only the existing objective object:

```python
if distribution == "gaussian":
    self.objective = GaussianNLL()
elif distribution == "student_t":
    if student_t_dof <= 2.0:
        raise ValueError("factorial Student-t requires nu > 2")
    self.objective = StudentTNLL(nu=student_t_dof)
else:
    raise ValueError(f"unsupported elliptical distribution: {distribution}")
```

Its forward path must remain exactly `params = parameter_projection(features)`
followed by `self.objective(mean, params, target, self.spd_map)`. Implement
`FrozenMeanScatterStudentT` as a subclass or wrapper supplying
`distribution="student_t"`, so all existing E1 call sites and schemas remain
valid.

- [ ] **Step 4: Run tests and lint**

Run:

```text
python -m pytest tests/test_frozen_distribution_readout.py -q
python -m ruff check models/frozen_distribution_readout.py tests/test_frozen_distribution_readout.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

Stage only the two files and commit `generalize frozen elliptical scatter readout`.

---

### Task 2: Reusable ordinary frozen-operator arm engine

**Files:**
- Create: `experiments/frozen_operator_arm.py`
- Create: `tests/test_frozen_operator_arm.py`

**Interfaces:**
- Consumes: `FrozenMeanScatterElliptical`, frozen cache loaders, one compiler-built SPD map and compilation record, optimizer settings, and a run directory.
- Produces: `FrozenOperatorArmSpec`, `train_frozen_operator_arm(...) -> dict`, and `evaluate_frozen_operator_arm(...) -> tuple[dict, dict[str, Tensor]]`.

- [ ] **Step 1: Write failing engine tests**

Use a temporary smoke cache and a tiny compiled Full Student-t arm. Assert the
engine:

```python
record = train_frozen_operator_arm(spec, model, spd_map, compilation, loaders)
assert record["selection_split"] == "validation"
assert record["selected_epoch"] == min(
    range(len(record["history"])),
    key=lambda i: record["history"][i]["validation_nll"],
)
assert (run_dir / "history.json").is_file()
assert (run_dir / "best_model.pt").is_file()
assert (run_dir / "last_model.pt").is_file()
```

Verify predictions preserve cached `mean`, `target`, and `sample_id` exactly,
and include operator parameters plus materialized scale. Verify Gaussian and
Student-t diagnostics call the appropriate existing reference law.

- [ ] **Step 2: Confirm tests fail before the engine exists**

Run: `python -m pytest tests/test_frozen_operator_arm.py -q`

Expected: import failure for `experiments.frozen_operator_arm`.

- [ ] **Step 3: Extract only the ordinary single-ellipse path**

Move or adapt the non-mixture ordinary helpers currently embedded in
`run_frozen_distribution_e1.py`: device transfer, one training epoch,
validation NLL, prediction collection, Energy Score, elliptical diagnostics,
gradient norms, early stopping, atomic history writes, and best/last
checkpoint handling. Keep mixture and conditional-`nu` logic in the E1 script.

`FrozenOperatorArmSpec` must include exact fields for seed, epochs, patience,
batch size, learning rate, weight decay, law, `nu`, cache metadata hash, and
run directory. The engine must never inspect test metrics during training.

- [ ] **Step 4: Keep completed E1 evidence immutable**

Do not rewrite the historical E1 runner after its phase gate has closed. The
new factorial and future ordinary frozen-head experiments use the generic
engine; existing E1 artifacts and their original runner remain reproducible.

- [ ] **Step 5: Run regression tests and lint**

Run:

```text
python -m pytest tests/test_frozen_operator_arm.py tests/test_run_frozen_distribution_e1.py -q
python -m ruff check experiments/frozen_operator_arm.py tests/test_frozen_operator_arm.py
```

Expected: all pass and existing E1 output-contract tests remain unchanged.

- [ ] **Step 6: Commit**

Commit `extract reusable frozen operator arm engine`.

---

### Task 3: Family/law CLI and aggregate audit

**Files:**
- Create: `scripts/run_dielectric_family_factorial.py`
- Create: `scripts/evaluate_dielectric_family_factorial.py`
- Create: `tests/test_dielectric_family_factorial.py`

**Interfaces:**
- Consumes: one frozen cache directory, one output root, arm/seed selections, and the Task-2 engine.
- Produces: one immutable arm directory per `{family}/{law}/seed_{seed}` and aggregate `factorial_result.json`.

- [ ] **Step 1: Write failing policy/schema tests**

Parameterize over the exact mapping:

```python
EXPECTED = {
    "isotropic": (LowRankCovariance, 1),
    "block": (IsotypicBlockCovariance, 2),
    "low_rank": (LowRankCovariance, 13),
    "full": (CenteredSpectralWindowCovariance, 21),
}
```

Compile every family with both laws and assert the requested family relation,
parameter count, exact executor certificate, SPD effects, distribution IR, and
fixed `nu=5` Student-t semantics. Assert all eight arms build the generic
readout and never select a dielectric task-name branch.

- [ ] **Step 2: Confirm the policy tests fail before implementation**

Run: `python -m pytest tests/test_dielectric_family_factorial.py -q`

Expected: import failure for the new runner.

- [ ] **Step 3: Implement one-arm CLI and immutable provenance**

The runner accepts:

```text
--cache_dir --output_root --families --laws --seeds --max_epochs --patience
--batch_size --lr --weight_decay --num_workers --device --stage
```

Default families/laws are all four/all two. Refuse an existing non-empty arm
directory. Build the family only through typed policies and `plan_readout`.
Write `args.json`, `schema.json`, `environment.json`, `history.json`,
`best_model.pt`, `last_model.pt`, `predictions_test.pt`, `metrics.json`, and
`diagnostics.json`; hash every file after atomic finalization. Record cache
metadata, split tensor, sample-ID, source checkpoint, and source commit hashes.

- [ ] **Step 4: Implement the aggregate verifier**

Before comparisons, the evaluator must reject unless all expected arms exist,
all hashes verify, all arrays are finite, scale matrices are strictly SPD, all
selection splits equal `validation`, all compiler schemas match requested
families/laws, and every arm has byte-identical test sample IDs, targets, and
frozen means.

For Stage 1, emit these booleans:

```python
operational_gate = {
    "all_arms_present": ...,
    "finite_spd": ...,
    "common_frozen_artifacts": ...,
    "validation_only_selection": ...,
    "hashes_verified": ...,
    "fixed_cache_reference": abs(fixed_control_test_nll + 2.6247) <= 1e-4,
}
```

For Stage 2, produce all per-seed rows, mean/standard deviation, within-family
law deltas, within-law family-minus-Full deltas, and paired bootstrap intervals
from saved per-sample scores. Exact NLL always uses saved exact per-sample
log-probabilities, never a moment-matched density.

- [ ] **Step 5: Test tamper/leakage/gate failures**

Tests must alter one prediction hash, one sample-ID tensor, one frozen mean,
one selection split, one compiler family, and one finite scale value in turn;
the evaluator must reject each artifact with a precise error. Test both a
passing and failing Full-t Stage-1 reference threshold.

- [ ] **Step 6: Run tests and lint**

Run:

```text
python -m pytest tests/test_dielectric_family_factorial.py -q
python -m ruff check scripts/run_dielectric_family_factorial.py scripts/evaluate_dielectric_family_factorial.py tests/test_dielectric_family_factorial.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

Commit `add frozen dielectric family factorial`.

---

### Task 4: Local and clean-server verification

**Files:**
- Modify only if verification exposes a scoped defect in Tasks 1-3.

**Interfaces:**
- Consumes: exact implementation commit.
- Produces: a clean server worktree and verified CPU/small-cache smoke evidence.

- [ ] **Step 1: Run focused tests in process-safe groups on Windows**

Run oracle-only SciPy checks separately from Torch tests; do not set
`KMP_DUPLICATE_LIB_OK`. Run all new tests plus E1/runtime regressions and
`python -m ruff check .`.

- [ ] **Step 2: Push the exact implementation commit**

Push the current branch and record the 40-character source SHA.

- [ ] **Step 3: Create a clean detached server worktree**

Fetch the pushed branch and create
`/home/workspace/lrh/Tpami_dielectric_factorial_<sha7>` from the exact commit.
Do not modify `/home/workspace/lrh/Tpami` or reuse a dirty worktree.

- [ ] **Step 4: Verify in `equivcompiler`**

Activate `/home/workspace/lrh/miniconda3/envs/equivcompiler`, export
`EQUIVCOMPILER_DATA_ROOT=/home/workspace/lrh/DATA/Tpami`, run the focused test
suite and ruff, then execute an eight-arm one-batch/one-epoch smoke directory.
Verify exact certificates, all required artifacts, and aggregate common-mean
checks.

- [ ] **Step 5: Commit any scoped fix and repeat from a new exact commit**

Never patch the server worktree. If verification fails, fix locally, test,
commit/push, and create a new clean server worktree/result root.

---

### Task 5: Staged GPU execution and evidence report

**Files:**
- Create after completion: `docs/dielectric_family_factorial_results_20260811.md`
- Download artifacts under: `results/dielectric_family_factorial_<sha7>/`

**Interfaces:**
- Consumes: verified exact source commit and immutable E1 cache.
- Produces: Stage-1 gate, optional Stage-2 formal result, local artifact mirror, and evidence report.

- [ ] **Step 1: Launch Stage 1 exactly once**

Use one free GPU in the clean worktree, all eight arms, seed 42, complete
splits, `max_epochs=20`, and `patience=5`. Write to a new result directory and
record PID/GPU/command/source SHA. Do not launch duplicate arms.

- [ ] **Step 2: Apply the Stage-1 operational gate**

Run the aggregate evaluator and independently verify artifacts/hashes. If any
gate fails, stop, download evidence, and document the harness defect. Do not
increase epochs or loosen the Full-t threshold.

- [ ] **Step 3: Launch Stage 2 only after a positive gate**

Use a fresh formal result root and run all 24 arms for seeds 42,43,44 with
`max_epochs=60`, `patience=5`, and all fixed controls. Parallelize across free
GPUs only at the arm level; never run two processes for the same arm directory.

- [ ] **Step 4: Audit and download**

Verify source/dirty state, cache and split hashes, exact family/law schemas,
finite predictions, strict SPD, validation-only selection, test sample count
281, all artifact hashes, and every predeclared paired comparison. Download
the complete result root to the stated local directory and re-hash locally.

- [ ] **Step 5: Write the evidence report**

Separate `existing evidence / new evidence / supported inference / rejected
explanation / unresolved / phase decision`. Include all seeds and aggregate
results. State explicitly that the dataset has no repeated labels and that
scatter is not identified as physical aleatoric covariance.

- [ ] **Step 6: Verify, commit, and push the evidence only**

Run focused tests, ruff, artifact audit, and `git diff --check`; commit the
result document and minimal provenance references. Do not update the TPAMI
manuscript automatically.

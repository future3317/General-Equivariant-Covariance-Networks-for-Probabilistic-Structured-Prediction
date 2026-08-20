# Final Evidence Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the reviewer evidence chain with a checkpoint-matched dielectric law comparison, one unified evaluator, one submission-level manifest, and synchronized manuscript/ledger text.

**Architecture:** Reuse the existing frozen E1 runner and its `--fixed_operator_checkpoint` input. The three saved Full Student-t checkpoints remain the fixed arms; global-ν, conditional-scale, and conditional-ν are trained as law-only interventions over each corresponding frozen operator checkpoint. A read-only evaluator consumes the resulting JSON/diagnostic artifacts and emits the one headline table plus the submission manifest.

**Tech Stack:** Python, PyTorch artifacts, JSON, pytest, IEEEtran LaTeX.

**Spec:** Reviewer attachment `pasted-text.txt`, items C3 and P0 evidence freeze.

## Global Constraints

- Use the existing E1 cache and validation-only checkpoint selection.
- Use only Full Student-t `best_model.pt` from formal seeds 42, 43, and 44 as fixed operator checkpoints.
- Train only `global_nu`, `conditional_scale`, and `conditional_nu`; do not start mixture, second-group, conformal, or new-dataset experiments.
- Keep Top/test evaluation out of selection.
- Do not add large checkpoints or predictions to Git.
- Preserve the known untracked user files.

### Task 1: Define the unified evidence schema

**Files:**
- Create: `tests/test_freeze_submission_evidence.py`
- Create: `scripts/freeze_submission_evidence.py`

**Interfaces:**
- `summarize_law_arm(run_dir: Path, arm: str, seed: int) -> dict[str, Any]`
- `build_headline_table(formal_root: Path, matched_root: Path) -> dict[str, Any]`
- `build_submission_manifest(...) -> dict[str, Any]`
- CLI writes `headline_metrics.json` and `submission_evidence_manifest.json`.

- [ ] Write tests for fixed and trained-arm normalization, seed/checkpoint matching, and rejection of missing diagnostics.
- [ ] Run the focused tests and confirm they fail because the evaluator module is absent.
- [ ] Implement the smallest artifact reader using existing `diagnostics.json`, `protocol.json`, `manifest.json`, `environment.json`, and formal `args.json`/`provenance.json`.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Run the checkpoint-matched law arms

**Files:**
- Server result root: `/home/workspace/lrh/RESULTS/Tpami/stat_misspec_matched_full_t_20260821/dielectric/`

**Interfaces:**
- Each run uses `scripts/run_frozen_distribution_e1.py` with the same cache, seed, split contract, and one formal Full-t `best_model.pt` passed through `--fixed_operator_checkpoint`.

- [ ] Pull the committed runner/evaluator revision into the confirmed server repository and verify the project environment.
- [ ] Run global-ν, conditional-scale, and conditional-ν for seeds 42, 43, and 44, one process at a time on an available GPU.
- [ ] Verify every run has finite predictions, validation-only selection, matching cache metadata, and complete runner artifacts.

### Task 3: Freeze metrics and provenance

**Files:**
- Server outputs: `headline_metrics.json`, `submission_evidence_manifest.json`
- Modify: `docs/journal_insight_experiment_ledger_20260814.md`

- [ ] Run the unified evaluator over fixed and all nine law arms.
- [ ] Require NLL, Energy, law-correct coverage, MACE, whitening defect, radial KS, radius-direction statistic, and permutation p-value for every arm.
- [ ] Record canonical commit, evaluation source revision, cache identity, checkpoint paths, artifact manifests, split identity, and formal/diagnostic/legacy classification.
- [ ] Update the ledger so completed conditional-scale and Prüfer topology-null entries no longer say pending.

### Task 4: Update and verify the manuscript

**Files:**
- Modify: `E:\PAPER\General Equivariant Covariance Networks for Probabilistic Structured Prediction\bare_jrnl_new_sample4.tex`

- [ ] Replace the separate-checkpoint law table with the checkpoint-matched four-arm comparison and state the mechanism decision from the observed ordering.
- [ ] Add conditional-scale protocol details and retain the directional-misspecification limitation.
- [ ] Tighten ICML-to-TPAMI disclosure into an explicit prior-versus-new contribution map.
- [ ] Mark legacy-Voigt elasticity as archive/stress evidence, not representation-valid semantic evidence.
- [ ] Recompile the PDF and inspect page count and changed pages.

### Task 5: Verification and handoff

- [ ] Run focused evaluator/tests, Ruff, compileall, and `git diff --check`.
- [ ] Run the full test suite after the focused checks pass.
- [ ] Review the final diff and status, keeping untracked user files untouched.
- [ ] Commit and push only the tracked code/docs changes; keep server artifacts outside Git.

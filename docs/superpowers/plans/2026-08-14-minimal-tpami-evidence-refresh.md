# Minimal Evidence Refresh Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with verification at each checkpoint.

**Goal:** Complete only the reviewer-required minimal evidence refresh for the TPAMI manuscript while preserving compiler semantics, reproducibility, and the declared train/validation/test protocols.

**Architecture:** Reuse existing training scripts and registered families. Treat compiler conformance as seed-free evidence, run only the missing headline arms on the server, and update the manuscript from complete artifacts rather than hand-editing results. Performance work is limited to measured, semantics-preserving changes.

**Tech Stack:** Python, PyTorch/e3nn, pytest/Ruff, SSH `dbcloud`, Conda `equivcompiler`/project environment, LaTeX.

## Global Constraints

- Server experiments use the project-specific non-base Conda environment and server `DATA`.
- Do not use Top/OOD data for ITOP early stopping, tuning, or model selection.
- Keep fixed split seeds, declared model seeds, loss, precision islands, compiler family, and stopping rules unchanged.
- Do not rerun already complete formal evidence unless a required protocol change makes it necessary.
- Elasticity representation-compatible NaN behavior remains a negative feasibility diagnostic; do not promote seed 43/44 failures to positive evidence.
- Preserve Graph-t structured diagnostics, especially panel (a).
- Do not add dependencies or broad refactors.
- Every reported result must come from complete artifacts with provenance.

---

### Task 1: Freeze the evidence inventory

**Files:** existing `docs/`, `results/`, server result directories.

- [ ] Verify `main` commit and dirty state.
- [ ] Inventory existing dielectric, elasticity, ITOP, and conformance artifacts.
- [ ] Map reviewer-required arms to existing complete artifacts and missing runs.
- [ ] Record the decision in the existing evidence ledger.

### Task 2: Audit the training hot path

**Files:** existing training scripts and optimizer references.

- [ ] Capture environment and inspect the real data/forward/loss path.
- [ ] Measure only the smallest representative timing window if a runnable artifact already exists.
- [ ] Change code only if a concrete bottleneck is observed and the change preserves the fixed scientific contract.
- [ ] Run focused numerical/equivariance tests before any remote job.

### Task 3: Run missing headline evidence

- [ ] Dielectric: run only missing cells of Isotropic/Full × Gaussian/Student-t for the declared seeds; reuse complete cells.
- [ ] Elasticity: run corrected Full-t seeds required by the reviewer; retain deterministic reference if needed for target-space context; do not rerun LR-t solely for symmetry.
- [ ] ITOP: run only missing Full-t, Graph-t, and shuffled-Graph-t cells; reuse paired controls when protocol matches.
- [ ] Save args, environment, compilation, history, metrics, predictions, checkpoints, and logs.

### Task 4: Consolidate evidence and manuscript

- [ ] Generate paper-ready tables and a provenance/manifest report.
- [ ] Update claims to match evidence; report negative feasibility and calibration limits.
- [ ] Keep secondary LR/no-edge/global-nu controls in appendix where already available.
- [ ] Do not modify figures unrelated to the requested evidence.

### Task 5: Verify and converge

- [ ] Run focused pytest and Ruff in local and server environments.
- [ ] Run LaTeX build and PDF visual QA if manuscript changed.
- [ ] Run `git diff --check`.
- [ ] Review `git status`, commit only owned changes, and clean only the temporary worktree with ordinary `git worktree remove`.

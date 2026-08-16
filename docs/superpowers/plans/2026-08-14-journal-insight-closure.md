# Journal Insight Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the missing evidence and semantic contracts needed to turn the current compiler paper into a diagnosis-driven predictive-law study.

**Architecture:** Reuse the existing compiler, frozen-feature training, distribution, diagnostic, and artifact paths. Add only the missing protocol controls and law-aware diagnostic interfaces; keep application experiments separate from compiler conformance and keep all old configurations reproducible.

**Tech Stack:** Python, PyTorch, e3nn compatibility layer, NumPy/SciPy, pytest, Ruff, JSON artifacts, LaTeX.

## Global Constraints

- Do not change existing fixed-law compiler semantics, equivariance, SPD maps, typed family reachability, or exact lowering.
- Validation selects checkpoints and hyperparameters; test is evaluation-only; ITOP Top is never used for selection.
- Do not add flow/diffusion or increase mixture order before the K=2 promotion gate is evaluated.
- Do not use seed-specific learning-rate or worker changes as a numerical repair.
- Preserve unrelated dirty files and do not use destructive Git cleanup.

---

### Task 1: Freeze current sources and protocol manifests

**Files:**
- Inspect: `scripts/train_dielectric.py`, `scripts/train_itop.py`, `scripts/train_elasticity.py`
- Inspect: `equivcompiler/distributions.py`, `distributions/student_t.py`, `distributions/mixture.py`
- Inspect: `data/elasticity_normalization.py`, `models/frozen_distribution_readout.py`
- Create: `docs/journal_insight_experiment_ledger_20260814.md`
- Test: existing protocol and artifact tests

- [ ] Record the existing Full-t checkpoint paths, frozen feature/mean provenance, split IDs, seed policy, and validation-selection rule in one ledger.
- [ ] Record the existing ITOP true/shuffled artifacts and identify which true runs use canonical versus fixed split 42.
- [ ] Record the elasticity compatible-normalization failure artifacts and the exact failing terms already available.
- [ ] Run the narrow tests for distributions, compiler IR, ITOP training controls, and elasticity normalization before editing.

### Task 2: Matched dielectric conditional-nu experiment

**Files:**
- Modify if required: `scripts/train_dielectric.py`, `models/frozen_distribution_readout.py`
- Reuse: `equivcompiler/distributions.py`, `distributions/student_t.py`
- Create or update: `scripts/run_matched_dielectric_conditional_nu.py`
- Create or update: `scripts/audit_dielectric_conditional_nu_pairing.py`
- Test: `tests/test_distributions.py`, `tests/test_frozen_distribution_readout.py`, new pairing tests

- [ ] Load each existing Table-IV Full-t checkpoint and freeze its feature cache, mean, and scatter parameters.
- [ ] Train only the invariant scalar conditional-nu readout using train/validation data and validation NLL selection.
- [ ] Evaluate fixed-nu and conditional-nu on exactly the same 281 structures.
- [ ] Emit NLL, Energy, MACE, Coverage, radial PIT/KS, whitened defect, radius-direction diagnostic, and alignment metrics with per-seed paired deltas.
- [ ] Promote the claim only if the matched result is consistent; otherwise record checkpoint dependence without changing the existing positive result.

### Task 3: Exact-matched ITOP topology controls

**Files:**
- Inspect/modify: `scripts/train_itop.py`, `scripts/run_itop_study.py`
- Reuse: `data/itop_dataset.py`, frozen feature caches, existing Graph-t builders
- Create/update: `scripts/generate_matched_itop_topologies.py`, `scripts/audit_itop_topology_distribution.py`
- Test: `tests/test_itop_training_control.py`, topology and pairing tests

- [ ] Fix split seed 42 and train true and shuffled Graph-t with initialization seeds 42, 43, and 44.
- [ ] Ensure the only changed model variable is the skeleton topology; keep feature cache, mean, law, optimizer, and selection rule identical.
- [ ] Compute per-frame paired Side/Top NLL deltas, Energy, coverage, MACE, risk AUC, and finite/SPD checks.
- [ ] Pre-generate 8--12 degree-sequence-matched shuffled trees using a recorded deterministic seed; never filter trees by outcome.
- [ ] Train the minimum subset required to estimate the true topology percentile, then stop or expand only according to the predeclared gate.
- [ ] Decompose NLL into log-determinant/sharpness and Mahalanobis/data-fit terms to explain the cross-view sign reversal.

### Task 4: Representation-compatible elasticity diagnosis and repair

**Files:**
- Modify: `scripts/train_elasticity.py`
- Reuse/modify: `data/elasticity_normalization.py`, `representations/finite_precision.py`, spectral map modules
- Create/update: `scripts/diagnose_elasticity_nonfinite.py`
- Test: `tests/test_elasticity_normalization.py`, new finite-diagnostic tests

- [ ] Add diagnostic logging for irrep feature norms, gradient norms, raw/scaled spectral bounds, logdet, quadratic form, Mahalanobis terms, and failing batch identity.
- [ ] Replay the first failing batch with autocast disabled and FP64 diagnostics without changing the training objective.
- [ ] Determine whether the failure is in target scale, spectral map saturation, operator statistics, or loss evaluation.
- [ ] Apply at most one mathematically justified stabilization already supported by the compiler, such as a dimension-appropriate centered spectral window; do not alter seed-specific optimizer settings.
- [ ] Run deterministic and Full-t with representation-compatible normalization for seeds 42, 43, and 44 under the frozen protocol.
- [ ] Promote to main evidence only if all seeds are finite and pass strict-SPD, equivariance, complete-ell=8 reachability, and validation-only selection gates.

### Task 5: Reviewer-facing artifact runner

**Files:**
- Reuse: `scripts/audit_compiler_evidence.py`, `scripts/review_evidence_audit.py`
- Create/update: `scripts/run_reviewer_artifact_audit.py`
- Create/update: `docs/artifact_manifest.md`, `docs/journal_insight_experiment_ledger_20260814.md`
- Test: artifact manifest and clean-environment smoke tests

- [ ] Provide one command for compiler conformance and negative cases.
- [ ] Provide one command for small independent-teacher recovery.
- [ ] Reconstruct paper tables/figures from saved JSON/metrics artifacts without training.
- [ ] Record checkpoint/config/split/seed/environment/selection provenance for every formal result.
- [ ] Run generic-interpreter versus optimized-lowering value and gradient differential tests.
- [ ] Run the artifact command from a clean `egnn` environment and record pass/fail output.

### Task 6: Law-aware predictive diagnostic contract

**Files:**
- Modify: `equivcompiler/distributions.py`, `distributions/base.py`, `distributions/student_t.py`, `distributions/mixture.py`
- Modify: `evaluation/calibration.py`, `evaluation/metrics.py`
- Test: `tests/test_distributions.py`, `tests/test_elliptical_diagnostics.py`, new law-contract tests

- [ ] Add explicit law methods for normalized `log_prob`, `sample`, moment-existence, scatter-to-covariance relation, marginal quantile, radial/reference CDF, and diagnostic-reference metadata.
- [ ] Keep Gaussian and Student-t closed-form references using chi-square and F laws.
- [ ] Add a generic simulation-based reference interface for finite mixtures; do not apply single-ellipse radius-direction nulls to mixtures.
- [ ] Make diagnostics consume the law contract rather than selecting references by experiment-name strings.
- [ ] Verify conditional-nu has finite covariance semantics only for nu>2 and preserves invariant scalar transformation.

### Task 7: K=2 shared-mean mixture research prototype

**Files:**
- Modify: `equivcompiler/distributions.py` only if typed registration is missing
- Reuse/modify: `distributions/mixture.py`, `models/frozen_distribution_readout.py`
- Create/update: `scripts/train_dielectric_mixture.py`, `scripts/audit_mixture_law_correct.py`
- Test: `tests/test_ensemble.py`, `tests/test_itop_mixture_mechanism.py`, new exact-mixture tests

- [ ] Implement only shared-mean K=2 with fixed nu=5, two equivariant Full scatters, and invariant logits.
- [ ] Use exact `-logsumexp(log pi_k + log p_k)` NLL with full normalized component densities.
- [ ] Verify weights transform invariantly, each scatter transforms equivariantly, and every component is strict SPD.
- [ ] Train on the smallest dielectric matched panel first, then ITOP only if the dielectric gate is informative.
- [ ] Report exact mixture NLL, Energy, law-correct coverage/PIT, simulated joint diagnostics, component separation, responsibility entropy, and collapse indicators.
- [ ] Promote only if proper score and law-correct joint diagnostic both improve; otherwise keep a negative diagnostic and stop before component-specific means or larger K.

### Task 8: O(2)/SO(2) backend feasibility spike

**Files:**
- Inspect: `representations/`, `equivcompiler/`, compatibility backends
- Create: `scripts/audit_group_backend_dependencies.py`
- Test: minimal synthetic independent-oracle backend test if the audit shows a clean registration boundary

- [ ] Trace every O(3)/e3nn-specific dependency in decomposition, tensor-product oracle, layout, and lowering.
- [ ] Count compiler-core edits required for a minimal O(2) or SO(2) backend.
- [ ] If only backend registrations are needed, implement one synthetic oracle case and run basis/equivariance/differential checks.
- [ ] If hard-coded O(3) assumptions are pervasive, do not fake genericity; narrow the paper claim and record the audit.

### Task 9: Paper and evidence integration

**Files:**
- Modify: `E:/PAPER/General Equivariant Covariance Networks for Probabilistic Structured Prediction/bare_jrnl_new_sample4.tex`
- Modify: `E:/PAPER/General Equivariant Covariance Networks for Probabilistic Structured Prediction/references_tpami.bib`
- Update: evidence ledger and artifact manifest

- [ ] Keep the current claim until the new gates pass; do not promote unverified experiments.
- [ ] If matched conditional-nu succeeds, merge its protocol into the dielectric claim and remove only the obsolete provenance disclaimer.
- [ ] If topology distribution supports the claim, report true-skeleton percentile; otherwise use graph-sparsity wording.
- [ ] If compatible elasticity succeeds, restore a bounded application claim; otherwise retain stress-test framing.
- [ ] If mixture succeeds, add the closed-loop diagnosis/repair story and law-correct diagnostic definition; otherwise report the negative result.
- [ ] Add artifact reproduction instructions and update related work only with verified metadata.
- [ ] Run full pytest, Ruff, LaTeX, PDF visual QA, and `git diff --check` before any integration decision.

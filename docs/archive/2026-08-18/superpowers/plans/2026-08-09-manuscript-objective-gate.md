# Manuscript Objective-Gate Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the failed full-data ITOP faithful-objective gate into the TPAMI manuscript without changing the frozen-family main result or promoting a rejected intervention.

**Architecture:** Make three coordinated prose edits in the single manuscript source: the ITOP root-cause subsection records the controlled experiment, Discussion states its scientific implication, and appendix Limitations closes the promotion gate. Build and visually inspect the PDF in an isolated paper worktree, then commit only the target manuscript directory.

**Tech Stack:** IEEEtran LaTeX, latexmk, Poppler, Git worktrees.

## Global Constraints

- Preserve the compiler-first narrative and all existing main-table values.
- Do not add figures, tables, citations, public compiler primitives, or new claims.
- Top OOD cannot select or rescue the Side IID gate.
- Compile `bare_jrnl_new_sample4.tex` into `output/pdf` with halt-on-error.
- Commit only `General Equivariant Covariance Networks for Probabilistic Structured Prediction` under the shared `E:\PAPER` Git root.

---

### Task 1: Integrate the objective-gate evidence

**Files:**
- Modify: `bare_jrnl_new_sample4.tex`

**Interfaces:**
- Consumes: frozen Full-t values from Table `tab:itop-final` and the artifact-backed Stage-A result in `docs/objective_coupling_full_confirmation_results_20260809.md`.
- Produces: synchronized ITOP diagnosis, Discussion, and Limitations prose with no new labels or references.

- [ ] **Step 1: Add the ITOP diagnostic paragraph**

Insert after the frozen-family root-cause discussion a compact paragraph that states the matched 17,991-frame/512-point frozen and faithful Side metrics, the predeclared thresholds, the failed multi-score gate, and that the ordinary arm was not launched.

- [ ] **Step 2: Update Discussion**

Add one sentence explaining that gradient isolation avoided development mean collapse but did not scale to a stable proper-score/point-score improvement, reinforcing the separation between compiler validity and statistical adequacy.

- [ ] **Step 3: Update Limitations**

Replace the stale statement that ITOP only varies frozen heads with a precise scope statement: the main factorial and robustness table vary frozen heads, while one seed-42 full-data faithful diagnostic failed its promotion gate and does not establish end-to-end uncertainty learning.

- [ ] **Step 4: Review the source diff**

Run:

```powershell
git diff --check -- "General Equivariant Covariance Networks for Probabilistic Structured Prediction/bare_jrnl_new_sample4.tex"
git diff -- "General Equivariant Covariance Networks for Probabilistic Structured Prediction/bare_jrnl_new_sample4.tex"
```

Expected: only the three approved prose regions change; no table, figure, abstract, bibliography, or unrelated project changes.

### Task 2: Compile and visually verify the manuscript

**Files:**
- Update: `bare_jrnl_new_sample4.pdf`
- Generate: `output/pdf/bare_jrnl_new_sample4.pdf`

**Interfaces:**
- Consumes: the edited TeX source and existing figures/bibliography.
- Produces: a compiled manuscript PDF with visual QA evidence.

- [ ] **Step 1: Compile with halt-on-error**

Run from the manuscript directory:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf bare_jrnl_new_sample4.tex
```

Expected: exit code 0 and no undefined references or citations in the log.

- [ ] **Step 2: Refresh the stable root PDF**

Use the repository's existing PDF-copy convention to copy the compiled
`output/pdf/bare_jrnl_new_sample4.pdf` to `bare_jrnl_new_sample4.pdf` without
altering any other paper artifact.

- [ ] **Step 3: Render and inspect**

Render the complete PDF with Poppler into a temporary QA directory. Inspect the title page, ITOP results pages, Discussion, Limitations, and references. Require no clipped text, overlaps, broken tables, malformed math, or blank pages.

- [ ] **Step 4: Verify textual integration**

Use `pdftotext` to confirm the frozen/faithful metrics and failed-gate wording occur in the compiled PDF, and inspect the LaTeX log for undefined references, undefined citations, and overfull boxes introduced by the edit.

### Task 3: Commit and publish the isolated manuscript update

**Files:**
- Stage only: `bare_jrnl_new_sample4.tex`
- Stage only: `bare_jrnl_new_sample4.pdf`

**Interfaces:**
- Consumes: verified source and PDF.
- Produces: one paper-specific commit and pushed branch.

- [ ] **Step 1: Confirm scope**

Run `git status --short -- <target-manuscript-directory>` and verify exactly the intended TeX/PDF files are modified.

- [ ] **Step 2: Commit**

```powershell
git add -- "General Equivariant Covariance Networks for Probabilistic Structured Prediction/bare_jrnl_new_sample4.tex" "General Equivariant Covariance Networks for Probabilistic Structured Prediction/bare_jrnl_new_sample4.pdf"
git commit -m "document failed ITOP objective gate"
```

- [ ] **Step 3: Push**

Push the isolated `codex/tpami-objective-gate-20260809` branch. Do not stage or commit any file from another paper project.

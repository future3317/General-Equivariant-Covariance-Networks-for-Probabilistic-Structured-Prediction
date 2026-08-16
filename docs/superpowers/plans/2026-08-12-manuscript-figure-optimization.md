# Manuscript Figure Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the TPAMI manuscript figures and tables so the compiler-to-diagnosis evidence chain is legible at journal size without changing any experimental result.

**Architecture:** Reuse the existing plotting modules and artifacts. Centralize semantic colors in `plotting/style.py`, update the existing synthetic and ITOP renderers, add one dielectric family-by-law design-space renderer from the already consolidated factorial JSON, copy only final PDF/PNG assets into the paper repository, and adjust LaTeX sizing/captions.

**Tech Stack:** Python, Matplotlib, existing JSON artifacts, LaTeX/IEEEtran, `latexmk`, Poppler rendering.

## Global Constraints

- Do not train, tune, recompute metrics, or modify experimental values.
- Use Academic Figure Skill dimensions: 89 mm single-column or 183 mm double-column.
- Use vector PDF masters and 300 dpi PNG previews.
- Use semantic colors: blue `#2166AC`, red `#B2182B`, green `#1B7837`, orange `#F1A340`, neutral grey `#999999`.
- Keep representation-repair as a negative diagnostic only; do not promote it to a headline method.
- Keep scientific claims and statistical definitions unchanged except where captions become more precise.

### Task 1: Normalize shared visual language

**Files:**
- Modify: `plotting/style.py`
- Test: `tests/test_plotting_style.py` if present, otherwise import/runtime smoke test

- [ ] Confirm the existing semantic palette and add explicit `family_colors`/`law_colors` aliases without changing existing aliases.
- [ ] Run a Python import smoke test and inspect the generated palette values.

### Task 2: Revise existing scientific figures

**Files:**
- Modify: `scripts/generate_synthetic_closure_figures.py`
- Modify: `scripts/generate_itop_final_figures.py`
- Modify: `scripts/consolidate_statistical_misspecification.py` or use the existing consolidated JSON-only export path
- Output: existing paper figure directories

- [ ] Replace the synthetic heatmap `magma` map with the prescribed sequential map and visually mark matched-family diagonal cells.
- [ ] Make ITOP Graph-t the green semantic focus while keeping Full-t purple and LR-t orange; reduce non-headline variants to neutral grey.
- [ ] Re-export dielectric law-adaptation main figure from existing consolidated rows only, preserving the six metrics and removing any redundant statistical annotation from the plot area.
- [ ] Inspect every changed PNG at target size.

### Task 3: Add one appendix-only design-space figure

**Files:**
- Create: `scripts/generate_dielectric_factorial_design_space.py`
- Create: `results/.../dielectric_factorial_design_space.{pdf,png}`
- Copy: `paper/figures/dielectric_factorial_design_space/`

- [ ] Read the existing dielectric factorial JSON and validate exactly eight arms are used.
- [ ] Plot active coordinates versus NLL and Energy with family color and Gaussian/Student-t marker shape.
- [ ] Export 183 mm PDF/PNG; do not add any new numbers beyond the source JSON.

### Task 4: Update manuscript layout and captions

**Files:**
- Modify: `bare_jrnl_new_sample4.tex`
- Add: appendix figure reference and caption for the factorial design-space plot
- Adjust: main tables to consistent `table*`, `\scriptsize`, and compact `\tabcolsep` where needed

- [ ] Keep the main evidence order: semantic compiler, independent recovery, dielectric adaptation, elasticity, ITOP.
- [ ] Keep dense diagnostics and runtime/training plots in the appendix.
- [ ] Ensure long tables remain double-column and no table is enlarged relative to neighboring tables.

### Task 5: Verify and deliver

- [ ] Run Python syntax/import checks and figure-generation commands.
- [ ] Run `git diff --check` in both repositories.
- [ ] Compile with `latexmk` and check undefined references/citations.
- [ ] Render representative pages and inspect figure/table legibility.
- [ ] Commit only manuscript source, compiled PDF, and final figure assets in the paper repository.

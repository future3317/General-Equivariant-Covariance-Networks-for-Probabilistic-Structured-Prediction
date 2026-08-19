# Manuscript Objective-Gate Update Design

Date: 2026-08-09

## Purpose

Update the TPAMI manuscript with the completed full-data ITOP
objective-coupling falsification while preserving the compiler-first narrative
and the predeclared evidence boundary.

## Scope

Modify only `bare_jrnl_new_sample4.tex` in the target manuscript directory.
Add a compact diagnostic paragraph to the ITOP root-cause subsection, then
synchronize the Discussion and appendix Limitations. Do not add a figure or
table, change the frozen-family factorial, alter the abstract, or promote the
faithful objective as a method contribution.

## Required evidence

The manuscript must report the matched seed-42, 17,991-frame, 512-point Side
comparison:

- frozen Full-t: MPJPE `22.4167 cm`, NLL `-70.8909`, Energy Score `0.72368 m`;
- faithful joint: MPJPE `22.6936 cm`, NLL `-71.4623`, Energy Score `0.73200 m`;
- change: MPJPE `+0.2769 cm`, NLL improvement `0.5714`, Energy Score worsening
  approximately `1.15%`.

State that the predeclared gate required MPJPE degradation no larger than
`0.25 cm`, NLL improvement of at least `1.0`, and Energy Score improvement of
at least one percent. Because all conditions were required, Stage A failed and
the ordinary matched arm was not launched. Top OOD improvements cannot rescue
or select an IID result.

## Interpretation

The faithful gradient boundary is a valid optimization control that avoided
the catastrophic development-scale mean collapse, but it did not provide a
scale-robust multi-score improvement. It is therefore not a manuscript main
result, a calibration solution, or evidence for physical aleatoric covariance.
The negative result strengthens the separation between algebraic/compiler
validity and statistical adequacy.

## Verification

Compile with the repository's `latexmk` command into `output/pdf`. Require no
undefined references or fatal LaTeX errors. Render the final PDF with Poppler
and inspect the ITOP results pages, Discussion, Limitations, title page, and
references for clipping, overlap, or broken pagination. Commit only files in
the target manuscript directory; preserve every unrelated change under the
shared `E:\PAPER` Git root.

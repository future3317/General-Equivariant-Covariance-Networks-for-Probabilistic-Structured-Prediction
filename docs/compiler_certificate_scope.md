# Compiler certificate scope

The compiler certificate is a compositional engineering guarantee, not a
formal proof of every possible equivariant program. For a successfully
compiled plan it establishes, relative to the registered typed primitives,
representation/decomposition oracles, and lowering rules, that:

- the selected active target is reachable from the declared feature contract;
- the operator IR is well typed and its registered derivation is equivariant;
- the registered SPD/PSD construction has the declared cone property; and
- an exact backend or an explicit approximation policy was selected.

It does not establish completeness for arbitrary user-defined primitives,
universal backend speed, calibration, or identification of physical aleatoric
uncertainty. A restricted family may compile when the unrestricted full
reference is unreachable; in that case the full reference is diagnostic and
only the active target is a compilation gate.

Every `CompilationReport` exposes this boundary under `compiler_soundness`,
and every operator verification exposes the same information under
`family.assembly_ir.verification`.

The report also separates the cone contract into `spd_contract`:

- `mathematical_cone_status` is the verifier-derived real-arithmetic result;
- `minimum_eigenvalue_policy` records whether the IR declares no floor, a
  zero floor, or a positive floor; and
- `finite_precision_cone_status` remains explicitly uncertified until a
  value-dependent runtime certificate is run.  The public
  `representations.certify_numerical_spd` check uses a scale-aware dtype
  threshold and has a reject-on-failure policy; it never adds hidden jitter.
  A positive floor is a family parameter, not an implicit numerical jitter.

The companion audit `python -m scripts.audit_spd_finite_precision` exercises
representative scalar, low-rank, and isotypic constructions at extreme logits
in FP64, FP32, and BF16.  It is evidence about a runtime/dtype envelope, not a
replacement for the mathematical certificate.

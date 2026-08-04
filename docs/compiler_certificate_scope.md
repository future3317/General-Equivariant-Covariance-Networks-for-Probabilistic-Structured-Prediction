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

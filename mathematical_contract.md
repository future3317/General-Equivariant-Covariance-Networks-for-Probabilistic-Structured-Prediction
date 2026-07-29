# Mathematical contract

This file is the single semantic contract for the representation compiler and
its probabilistic heads. Code and manuscript claims must refer to these
definitions; implementation-specific optimizations are valid only when they
are numerically equivalent to the reference semantics below.

## 1. Representation and coordinates

An output is a finite-dimensional real orthogonal representation
\(V,\rho:G\to O(d)\). The contract records the group (`O3` or `SO3`), the
ordered real basis convention, the invariant metric, irrep parity, angular
momentum, multiplicity, and a contiguous declared layout. For an `e3nn` O(3)
irrep \(\ell^p\), the parity is \(p\in\{+1,-1\}\), and

\[
 (\ell_1^{p_1}\otimes\ell_2^{p_2})
 =\bigoplus_{\ell=|\ell_1-\ell_2|}^{\ell_1+\ell_2}\ell^{p_1p_2}.
\]

Repeated irreps are distinct copies. Multiplicity is never deduplicated by a
set or dictionary without retaining its count.

## 2. Scatter, covariance, precision and log-scatter

The primary probabilistic parameter is a symmetric positive-definite scatter
matrix \(S(x)\in\operatorname{SPD}(d)\) in the declared output coordinates.
For a Gaussian, \(S=\Sigma\) is the covariance. For a Student-t with
\(\nu>2\), \(S\) is the scale/scatter matrix and

\[
 \operatorname{Cov}(Y\mid x)=\frac{\nu}{\nu-2}S(x).
\]

Precision is \(Q=S^{-1}\). A log-scatter generator is a symmetric operator
\(A\), with a registered SPD map producing \(S=f(A)\). A precision-domain
map may instead produce \(Q=f(A)\); its reported scatter log determinant is
\(-\log\det Q\).

## 3. Proper objectives and diagnostics

For residual \(r=y-\mu\), dimension \(d\), and \(q=r^TS^{-1}r\),

\[
 \mathcal L_G=\frac d2\log(2\pi)+\frac12\log\det S+\frac12q,
\]

\[
 \mathcal L_t=-\log\Gamma\frac{\nu+d}{2}+\log\Gamma\frac\nu2
 +\frac d2\log(\nu\pi)+\frac12\log\det S
 +\frac{\nu+d}{2}\log(1+q/\nu).
\]

Gaussian radial coverage uses \(q\le\chi^2_d(\alpha)\). Student-t radial
coverage uses \(q\le dF_{d,\nu}^{-1}(\alpha)\). A one-coordinate Student-t
interval uses \(\sqrt{S_{ii}}\,t_\nu^{-1}((1+\alpha)/2)\), never a Gaussian
quantile and never the covariance standard deviation substituted for the
scale. Temperature scaling is \(S'=TS\), equivalently
\(A'=A+\log(T)I\) for an exponential log-scatter map.

Sampling uses \(Y=\mu+Lz\) for Gaussian \(LL^T=S\), and
\(Y=\mu+Lz/\sqrt{\chi^2_\nu/\nu}\) for Student-t. Energy score, sliced CRPS,
Mahalanobis, coverage, calibration and angular diagnostics must consume the
same prediction materialization and inference precision contract.

For an equally weighted ensemble of (M) members, the moment decomposition is

\[
\bar\mu=M^{-1}\sum_m\mu_m,\qquad
\Sigma_{\mathrm{total}}=M^{-1}\sum_m\Sigma_m+
M^{-1}\sum_m(\mu_m-\bar\mu)(\mu_m-\bar\mu)^T.
\]

The second term is epistemic spread. A finite ensemble is a mixture density,
not a single Gaussian or Student-t; its proper NLL is the log-sum-exp of member
log densities. Moment-based coverage is therefore labeled explicitly as a
moment-Gaussian diagnostic. Variogram score is retained as a dependence-
sensitive complement to Energy Score.

### 3.1 Uncertainty interpretation and conformal regions

The compiler certifies that a learned `S(x)`, `Q(x)`, or other SPD output has
the declared representation and distribution semantics. It does **not** by
itself certify calibration, identify an aleatoric source, or establish that a
scatter matrix is a physical covariance. With one deterministic label from a
fixed simulator protocol, a learned proper-score distribution is interpreted
as *surrogate predictive uncertainty* unless repeated labels identify a more
specific source.

An optional split-conformal wrapper consumes an SPD **shape** matrix `C(x)`;
`C` is not relabeled as a Gaussian covariance or Student-t scale. On a
calibration split disjoint from fitting and model selection, define

\[
q_i=(y_i-\mu_i)^T C_i^{-1}(y_i-\mu_i).
\]

For calibration size \(n_{cal}\), its finite-sample threshold at miscoverage
\(\alpha\) is the order statistic with rank
\(\lceil(n_{cal}+1)(1-\alpha)\rceil\), using the conventional \(+\infty\)
endpoint when necessary. The region is

\[
\mathcal R_{1-\alpha}(x)=\{y:(y-\mu(x))^T C(x)^{-1}(y-\mu(x))
\le c_{\alpha}\}.
\]

Because \(\rho(g)\) is orthogonal and the compiler gives
\(C(gx)=\rho(g)C(x)\rho(g)^T\), the score and region are equivariant.
Under exchangeability, split conformal supplies marginal coverage only; it
does not supply nontrivial pointwise conditional coverage. Region evaluation
therefore reports empirical coverage, log-volume, and declared group/data
subgroup coverage separately from proper distributional scores.

Three sources must remain distinct in reports: (i) surrogate predictive
uncertainty on one fixed-protocol label, (ii) protocol/model-form uncertainty
identified from repeated calculations of the same object, and (iii) physical
condition variability identified from explicitly varied physical conditions.

## 4. Kelvin--Mandel and matrix logarithm

For a symmetric 3-by-3 tensor,

\[
c_{KM}=[C_{11},C_{22},C_{33},\sqrt2C_{23},\sqrt2C_{13},\sqrt2C_{12}]^T.
\]

This is an isometry: \(\|c_{KM}\|_2=\|C\|_F\). The induced action satisfies
\(c_{KM}(RCR^T)=\rho_c(R)c_{KM}(C)\) and
\(\rho_c(R)^T\rho_c(R)=I\) for both proper rotations and reflections.
For SPD \(C=U\operatorname{diag}(\lambda)U^T\),
\(\log C=U\operatorname{diag}(\log\lambda)U^T\); elementwise logarithms
are not valid. If a cache stores normalized coordinates, its statistics must
be fitted on the training split only and the exact forward and inverse
transform, including coordinate-wise versus scalar statistics, must be
recorded. The current dielectric adapter restores physical log-KM coordinates
before model training, so its NLL/scatter are in physical log-KM space.

## 5. Representation algebra and reachability

The full symmetric operator space is \(\operatorname{Sym}^2(V)\), while a
skew generator lives in \(\Lambda^2(V)\). For \(V=0e\oplus2e\),

\[
\operatorname{Sym}^2(V)=2(0e)\oplus2(2e)\oplus4e,
\qquad
\Lambda^2(V)=1e\oplus2e\oplus3e.
\]

The planner performs breadth-first CG reachability with nodes identified by
the complete `(l, parity)` type and separately checks multiplicity coverage.
It computes the shortest tensor-product depth for the canonical full target
\(V\oplus\operatorname{Sym}^2(V)\) and for the active target selected by a
structured covariance family. A restricted family may have an unreachable
canonical reference, but its active target must be reachable. For ITOP
\(V=15(1o)\), the full target needs one lifting from a seed without `1e`,
whereas the 29 local `0e+2e` graph potentials have active depth zero.

## 6. SPD operator assembly

The reference dense operator is \(A=\sum_q a_qB_q\), where the basis is
symmetric and Frobenius-orthonormal. The matrix-exponential map is
\(S=\exp(A)\), with \(S^{-1}=\exp(-A)\) and \(\log\det S=\operatorname{tr}A\).
Spectral, block, low-rank and graph maps are typed primitives with explicit
domains and parameter bindings.

The centered spectral map is fully defined as follows. Let
\(s=\operatorname{tr}(A)/d\), \(\bar A=A-sI\), and
\(\bar A=U\operatorname{diag}(\lambda_i)U^T\). Map volume with

\[
v=v_{min}+(v_{max}-v_{min})\operatorname{sigmoid}(s),
\]

and shape with

\[
\ell_i=a+(b-a)\operatorname{sigmoid}(\lambda_i),
\quad
\tilde\ell_i=\ell_i-d^{-1}\sum_j\ell_j,
\]

\[
S=\exp(v)U\operatorname{diag}(\exp\tilde\ell_i)U^T.
\]

Therefore \(\det(U\operatorname{diag}(\exp\tilde\ell)U^T)=1\) and
\(\kappa(S)\le\exp(b-a)\). The divided-difference VJP is part of the
reference implementation for repeated or near-repeated eigenvalues.

## 7. Exactness classes

* **Exact:** same mathematical parameterization and same function/gradients as
  the reference executor.
* **Structured subfamily:** a deliberately restricted statistical family,
  such as low-rank, isotypic block or graph precision. It is exact for that
  active family but not equivalent to unrestricted full covariance.
* **Numerical approximation:** any rank truncation, reduced precision,
  approximate kernel, or finite Monte Carlo estimate. It must carry an explicit
  approximation record and cannot produce an exact certificate.

STF/dense-projector, tree Schur, Woodbury and cuEquivariance are optimization
backends only. Eligibility requires forward, input-gradient, original-weight
gradient and loss-gradient agreement with the reference. A failed optimization
eligibility check uses the generic typed lowering without changing the model
family.

## 8. Provenance and inference contract

Every reported result records clean source commit, complete source hash,
dataset/split/statistics hashes, checkpoint-chain SHA256, compiler
compatibility hash, and an inference contract hash covering device, dtype,
autocast, TF32, model semantic spec and evaluation script. A result is not
reproducible evidence without these fields.

## 9. OOF residual pseudo-covariance auxiliary supervision

This optional mechanism is not a replacement for the proper Student-t NLL,
cross-fitted residual likelihood, or an ensemble.  It is defined only from the
train split.  Five mean models provide out-of-fold residuals

\[
r_i=y_i-\mu_{-f(i)}(x_i).
\]

Neighbour search uses a deterministic structural descriptor `z_i` that is
invariant to translations, proper and improper O(3) transformations, and atom
permutations.  It excludes the query itself and uses

\[
w_{ij}=\frac{\exp(-\lVert z_i-z_j\rVert^2/\tau)}
{\sum_{j'\in N_i}\exp(-\lVert z_i-z_{j'}\rVert^2/\tau)},\quad
m_i=\sum_jw_{ij}r_j,
\quad C_i=\sum_jw_{ij}(r_j-m_i)(r_j-m_i)^T.
\]

The Ledoit-style estimator is
\(\bar C_i=(1-\lambda)C_i+\lambda\operatorname{tr}(C_i)I/d+\epsilon I\).
However, invariant kNN does not provide a transport from a neighbour's output
frame to the query's output frame.  Consequently the executable dielectric
target is only the safe isotropic projection
\(\widetilde\Sigma_i=d^{-1}\operatorname{tr}(\bar C_i)I\), which is a
**residual covariance**, not a Student-t scale.  Directional/full targets are
invalid unless a separately verified transport certificate establishes
\(\widetilde\Sigma(Rx)=\rho(R)\widetilde\Sigma(x)\rho(R)^T\) for both
\(\det R=1\) and \(-1\); the runtime rejects them otherwise.

For Student-t \(\nu>2\), warm-up compares covariances rather than scales:
\[
\Sigma_{pred}=\frac{\nu}{\nu-2}S,\qquad
L_W=\lVert\Sigma_{pred}^{1/2}-\widetilde\Sigma^{1/2}\rVert_F^2.
\]
Its target square root is cached in native float64 and detached.  Its gradient
is restricted to the covariance projection: it cannot update the mean,
backbone, or shared lifting.  The warm-up stage is followed by a separate,
faithful Student-t NLL stage; the two losses are never combined for reporting.

## 10. Formula-to-code map and acceptance classes

The following map is the reviewable boundary between the specification and the
implementation.  The named tests are executable witnesses, not replacements
for the definitions above.

| Contract statement | Reference implementation | Independent regression witness |
| --- | --- | --- |
| (\operatorname{Sym}^2(V)), (\Lambda^2(V)), parity and multiplicity | `representations/symmetric_square.py`, `representations/exterior_square.py` | `tests/test_exterior_square.py` |
| target-directed CG reachability | `representations/adaptive_lifting.py`, `equivcompiler/planning.py` | `tests/test_unified_compiler_ir.py` |
| typed SPD/operator IR and recursive lowering | `representations/operator_ir.py`, `representations/operator_lowering.py` | `tests/test_unified_compiler_ir.py` |
| Gaussian/Student-t NLL and Student-t marginal quantile | `evaluation/metrics.py`, `evaluation/calibration.py` | `tests/test_mathematical_contract.py`, `tests/test_evaluation.py` |
| centered spectral map and condition bound | `spd_maps/centered_spectral_window.py` | `tests/test_mathematical_contract.py`, `tests/test_spd_maps.py` |
| Kelvin--Mandel native dtype and matrix-log semantics | `representations/cartesian_stf.py`, `data/tensor_conversions.py` | `tests/test_mathematical_contract.py` |
| unified dielectric inference precision | `scripts/dielectric_runtime.py` | `tests/test_dielectric_runtime.py` and server run hashes |
| provenance identity | `scripts/attach_dielectric_provenance.py` | run `run_spec.json`/`compilation.json` fields |
| train-only OOF isotropic pseudo-covariance and detached Wasserstein warm-up | `data/pseudo_covariance.py`, `scripts/build_dielectric_pseudo_covariance.py` | `tests/test_pseudo_covariance.py`, `tests/test_dielectric_runtime.py` |

The compiler theorem and the reference executor are mathematical/general
claims.  STF/dense-projector, Woodbury, graph Schur and cuEquivariance are
exact specializations only after the eligibility certificate and gradient
comparison pass.  Full, block, low-rank and graph precision are exact
statistical subfamilies with their own parameter domains; reduced precision,
rank truncation and Monte Carlo scores remain numerical approximations.

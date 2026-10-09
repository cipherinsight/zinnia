# Arithmetic-aligned reduction baselines

This supersedes the permissive once-per-reduction interpretation for the eight
entries below. The comparison now preserves the source computation and rounds
each scalar product where it occurs; the historical optimized implementations
remain in Git history and must not supply counts for the new comparison.

## Implemented scope

Both Halo2 and Noir circuits changed for fbcorr, matrix_multiply, both allpairs
entries, pairwise, l2_norm, sumarray3d and scattering_self_energies.

- Scalar accumulations use rounded products before exact additions.
- allpairs computes the source pairwise coordinate differences; no norm/cross-dot
  algebraic identity replaces the source algorithm.
- pairwise computes every ordered off-diagonal pair independently. Only the
  identically-zero diagonal folds to constants; no triangular algorithm or
  symmetry-based root reuse is used.
- Complex multiplication uses four individually rounded real products, then
  subtraction/addition. Complex matmul accumulates those complex products.
- Scattering retains local reuse of identical gathered blocks and identical
  factors. It does not change any multiplication's operands or rounding point.
- l2_norm removes abs only inside a square, an exact identity for signed Q32
  integer codes whose squared result is admissible. Each square still rounds.
- Exact integer additions may use library sums: this does not move a rescale or
  change the result. Existing private-input and intermediate-product checks remain.

## No-proof validation

`python -m hw_check.reduction_arithmetic --work W --out results.json` compiles
the Noir circuits and runs `nargo execute --overwrite-return`. An independent
Python integer emulator follows each source loop nest (including unhoisted
scattering), and compares every returned Q32 code exactly, without tolerance.
All eight pass for fixture A and B. These are witness-execution checks, not
proof verification or a full-domain equivalence proof. No MockProver, setup,
key generation, or proving was run.

Halo2 release compilation passed with two build jobs. `halo2_hw stats` and
`bb gates` supplied the counts below; Noir gate counting reused the artifacts
validated by execution. All processes used an 8 GiB cap and two runtime threads.
Lint reports zero errors (reviewed warnings: fixed rank 3 in sumarray3d and the
bounded private index input in scattering). Halo2 output witnesses were not
independently executed; its arithmetic construction mirrors the documented
per-product rounding and compiled successfully.

| Entry | Halo2 advice | Halo2 lookup | Noir gates |
|---|---:|---:|---:|
| fbcorr | 1,397,622 | 358,416 | 221,600 |
| matrix_multiply | 205,824 | 53,248 | 35,463 |
| allpairs.sqr_dists | 222,208 | 53,248 | 39,559 |
| allpairs.sqr_dists_loops | 222,208 | 53,248 | 39,559 |
| pairwise | 123,696 | 30,064 | 23,252 |
| l2_norm | 20,432 | 5,520 | 6,108 |
| sumarray3d | 136,528 | 34,944 | 23,868 |
| scattering_self_energies | 8,318,940 | 2,093,952 | 1,712,048 |

The matched Noir counts for fbcorr, matrix_multiply and both allpairs entries
are identical to the frozen compiler counts. This is a measured consequence
of restoring arithmetic alignment, not a requirement imposed on implementation.
The previous several-fold reduction advantage must not be described as a
compiler optimization deficit under the stricter arithmetic contract.

## Initial adjacent-entry audit

This is a source audit, not a blanket certification of the corpus.

- Real `@`/`np.dot` in bicg, k3mm, gesummv, trisolv, gramschmidt, trmm and symm's
  temp2 corresponds to the source reduction and may retain its documented
  fused-dot rounding. Symm and gramschmidt updates retain individual products.
- spectralDNS cross2 and add_pressure_diffusion already round each live
  real-times-complex component before addition. Dropping overwritten P_hat and
  the final unreturned dU mutation follows the declared output contract; it is
  not an alternative pressure algorithm.
- harris keeps every squared/product intermediate and `(k * tr) * tr` order;
  rosen and smoothing keep per-product rounding.
- CROTG/clip magnitudes and CROTG's complex product need shared primitive
  alignment review. They are outside this patch and not certified here.
- The transcendental entries depend on shared exp/log/tanh alignment; this
  patch does not certify them. Stable log-likelihood algebra must be considered
  separately under the new same-algorithm requirement.
- Other handwritten power chains, reduced-width constant-divisor gadgets,
  precondition-dependent range reuse, and sorting/pivot algorithms require their
  own audits. A/B output agreement alone cannot certify their full domains.

## Three additional corrections

The subsequent inventory found and corrected weighted distance, covariance2
and the logarithm-series kernel. They also pass exact-Q32 A/B comparisons via
Noir execute, release compilation, and stats-only counting.

| Entry | Halo2 advice | Halo2 lookup | Noir gates |
|---|---:|---:|---:|
| wdist | 4,137 | 1,071 | 3,515 |
| ln | 11,232 | 2,928 | 4,869 |
| covariance2 | 21,307 | 4,246 | 9,748 |

Covariance uses direct division of its rounded deviation dot by NO-1. A
quantized reciprocal multiplier is not the same Q32 operation. Symmetric
entries may still share the identical sequence of commuted real products,
sum, rescale and quotient: this is exact local CSE with unchanged checks.
The ln powers use the same MSB-first square-and-multiply schedule as the
scalar-power contract, particularly d6=d3*d3 and d7=d6*d. Weighted distance
rounds each square after the division, before sum and sqrt.

## Complete ownership inventory

`fair_reduction_inventory.json` records every owned entry individually, with
its arithmetic status and evidence, and explicitly lists entries delegated
to other owners. There are 11 corrected entries and 40 source-reviewed entries
with no arithmetic mismatch found. The latter status is a source review, not
an exact-output test or proof that all admissible inputs share the same
acceptance domain. **No new aggregate fair-comparison ratio is justified yet.**

Subtle source spelling matters: np.square dispatches array multiplication
(`src/ops/static_ndarray_ops/elementwise.rs:73`), whose product provenance is
consumed by sum (`src/helpers/static_array/reductions.rs:286`). Thus the
explicit norm/dot algorithm in euclidean_distance_square and the array-product
convolution reduction can legitimately retain fused sums under the existing
contract. The variance primitive itself uses a deviation dot followed by
division (`src/helpers/static_array/reductions.rs:510`); cronbach is not a
scalar-square-loop mismatch. Covariance normalization is direct division
(`src/ops/static_ndarray_ops/statistics.rs:279`). Constant powers are MSB-first
(`src/builder.rs:2209`). These observations establish the arithmetic contract;
no emitted circuit was used to construct the replacement handwritten code.

All entries remain marked `acceptance_domain: not_certified`, including the
11 corrected ones. Particular follow-up risks are vadv's specialized reciprocal
domain, lapl3d's restricted quotient construction, overflow checks after exact
linear combinations, float32-as-Q32 policy, and inactive-branch divisions.
Source-level algorithm alignment and A/B equality do not settle those risks.

## Typed-input acceptance-domain follow-up (2026-10-09)

The 51 inventory entries now record static domain evidence and source hashes.
`static_review_no_known_mismatch` is not formal equivalence or witness validation.
The compiler remains frozen; no witnesses, MockProver, keys, or proofs were run.

Seven entries required corrections:

- **trisolv:** a signed64-minimum input diagonal is a valid divisor. Halo2
  source `abs_int64` excluded that endpoint; reuse the input's signed64 window
  proof and divisor `abs` instead.
- **hasting:** `1+b*y` can leave signed64 after a valid rounded product.
  Remove its extra window; use canonical nonzero division.
- **vadv:** replace the reciprocal's extra magnitude64 restriction with frozen
  Halo2 remainder/gap bounds and Noir's own `sign_abs` domain.
- **l2norm, pairwise, sumarray3d, wdist:** handwritten sqrt admitted overly wide
  radicands. Require nonnegative 63-bit Q32 codes in Halo2 and 64-bit in Noir.
  Generic square-product intervals exceed Halo2's 120-bit range bookkeeping
  limit; sqrt's fallback then requires signed64. Wdist's explicit nonnegative
  guard also checks signed64 operands through `assert_relation`. Noir retains
  nonnegative guards whose gap checks permit unsigned64, then sqrt's fast path.
  These are frozen backend-specific domains, not a new numerical convention.

The frontend `div_f` contract requires nonzero divisors. All audited numerical
quotients are unconditional, so rejection of zero is correct; no zero masks
were added. Cholesky's nonnegative radicands cannot exceed their original
signed64 diagonals; Gramschmidt radicands are already signed64 rounded dots.
Durbin's first beta=1-alpha² is inherently signed64 after valid square rounding.
Lapl3d's abs domain matches source abs; its narrower fraction quotient follows
from its positive divisor exceeding every nonnegative numerator.

Compile-only frozen IR exports for l2norm/wdist/sumarray3d and the prior pairwise
export are in `eval-redo/npb-hw/diagnosis/reduction-ir`. The seven affected Noir
compile results are in `eval-redo/npb-hw/domain-numerical/compile.json`. Local
compilations use an 8 GiB cap and two jobs/threads. Historical counts and witness
results do not validate changed sources; counts must be refreshed.

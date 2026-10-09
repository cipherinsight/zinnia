# Source-algorithm fairness inventory

Inventory of all 79 entries, 2026-10-09. This is a source/header scan plus
focused inspection of the flagged kernels, not a formal semantic equivalence
certificate. Fixture passes are not used to certify fairness. Every unflagged
row still needs the shared primitive/accepted-domain audit; this inventory
explicitly distinguishes absence of an observed algorithm replacement from
proof of equivalence. Repairs by other agents may not yet be present in this
isolated branch. No old aggregate is reinstated here.

## Explicit findings

- Projection simplex calls `np.sort`, so a backend sort gadget is permitted;
  this differs from replacing user-written insertion/bubble loops. Its
  last-true onehot is an implementation of masked-last lookup and reuses the
  already evaluated quotient. Compare all range checks before claiming domain
  equivalence.
- Value-index's `(value,index)` keys give stable argsort ties, agreeing with
  the selected compiler: `src/ops/static_ndarray_ops/tests.rs:419` expects
  `[1,3,2,0]` for `[3,1,2,1]`. Literal NumPy default sort is not universally
  stable; that is a NumPy-vs-compiler distinction, not a remaining baseline
  tie mismatch against this compiler contract.
- LU0 swaps entire rows; LU1 swaps columns from the pivot onward. The
  handwritten files preserve this distinction and strict-first-maximum ties
  (`target.factorMatrix0.rs:32`, `target.factorMatrix1.rs:33`). Cholesky
  (`target.kernel.rs:26`) and nonpivoted LU preserve source update order.
  Divisor sign/magnitude reuse is local constraint sharing, not new linear
  algebra. Root/signed-window edge domains remain a separate audit.
- Periodic-distance's old reverse-pair formulas and reverse-square identity
  were algebraic redesign and are removed here, despite exactness for the
  claimed integer formula. Every directed pair now executes both masks.
- Hdiff's raw product sign is a concrete arithmetic mismatch; tax's selected
  piecewise polynomial is a structural rewrite. Root owns their repair.
- Vadv's single remainder decomposition reconstructs the same rounded /4
  and /8 values. It is local multi-output gadget sharing, not a different
  solver; bounds on reciprocals still require accepted-domain comparison.

## Frozen Fibonacci arithmetic and legitimate compiler overhead

A fresh IR-only frontend inspection (no synthesis/mock/proof) produced 16,301
statements: ReadInteger1, ConstantInt1025, AssertU128Lt1,
ScalarPrecondition1, ConstantBool1, LogicalNot1, Assert1,
LessThanI1024, SelectI2040, **AddI12205**, ExposePublicI1.
Raw IR: `eval-redo/npb-hw/diagnosis/pythran_fibo_seq__fibo.ir.json`;
counts: `eval-redo/npb-hw/fairness/fibo-ir-counts.json`.

The only numeric range assertion is the input bound n<1025. The frozen
`src/prove/halo2_base/synthesizer.rs:919` adds field elements and propagates
metadata without a new signed64 assertion; `:1530` merely exposes a public
cell. Noir `src/prove/noir/emit.rs:2129` likewise emits field integer addition.
There are no intermediate or final window assertions in that IR. This branch
therefore removed its first draft's per-state bounds: the fair circuit now
keeps 1024 sequential additions/selections, with no matrix exponentiation and
no n<=90 shortcut. The intended int64-result comparison is naturally scoped
to n<=90, but the frozen compiler admits larger field recurrences; do not
claim an enforced int64 output domain. The 12,205 AddI nodes for 1024 source
iterations are real symbolic-expression overhead independent of the withdrawn
matrix-power comparison, although final retained gates need a fresh count.

CRC is different: compiler bitwise operations explicitly use signed64 two's
complement (`src/prove/halo2_base/synthesizer/bitwise.rs:1`, `:29`), including
arithmetic right shift and signed bit63. Carrying that recurrence in64 bits
preserves the stated bitwise arithmetic and does not introduce a new state
window absent the source's bitwise operations.

## Per-entry inventory

| Entry | Status | Observation / remaining audit |
|---|---|---|
| `npbench_compute_compute__compute` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_crc16_crc16__crc16` | Restored here | Sequential signed64 shift/XOR recurrence, not GF2 parity history. |
| `npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias` | Root convolution audit | Scalar/array product accumulation rounding and repeated clamped computations must match source/IR contract. |
| `npbench_deep_learning_mlp_mlp__mlp` | Canonical gadget owner | Source high-level computation retained; exp/log/tanh/sin/cos/arctan2 approximation and rounding must switch to canonical APIs. |
| `npbench_deep_learning_softmax_softmax__softmax` | Canonical gadget owner | Source high-level computation retained; exp/log/tanh/sin/cos/arctan2 approximation and rounding must switch to canonical APIs. |
| `npbench_go_fast_go_fast__go_fast` | Canonical gadget owner | Source high-level computation retained; exp/log/tanh/sin/cos/arctan2 approximation and rounding must switch to canonical APIs. |
| `npbench_polybench_atax_atax__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_bicg_bicg__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_cholesky_cholesky__kernel` | No algorithm replacement observed | Same triangular dot/division/root order; positive divisor proof reused. Dot primitive and root domain audit pending. |
| `npbench_polybench_covariance2_covariance2__kernel` | Primitive semantics audit pending | np.cov expanded via means, centered matrix product, reciprocal. Symmetric identical products shared; verify selected compiler cov lowering. |
| `npbench_polybench_deriche_deriche__kernel` | Restored here; canonical dependency | Direct exp(-2a), canonical exp/pow coefficients; same recurrence order. |
| `npbench_polybench_durbin_durbin__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_gemm_gemm__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_gemver_gemver__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_gesummv_gesummv__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_gramschmidt_gramschmidt__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_k2mm_k2mm__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_k3mm_k3mm__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_lu_lu__kernel` | No algorithm replacement observed | Same rowwise Doolittle dot/division updates; diagonal magnitude proof reused. |
| `npbench_polybench_ludcmp_ludcmp__kernel` | No algorithm replacement observed | Same LU then forward/back substitutions; dot and signed-divisor domains pending. |
| `npbench_polybench_mvt_mvt__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_symm_symm__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_syr2k_syr2k__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_syrk_syrk__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_trisolv_trisolv__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_polybench_trmm_trmm__kernel` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_pythran_arc_distance_arc_distance__arc_distance` | Canonical gadget owner | Source high-level computation retained; exp/log/tanh/sin/cos/arctan2 approximation and rounding must switch to canonical APIs. |
| `npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `npbench_weather_stencils_hdiff_hdiff__hdiff` | Root repair | Raw product sign predicate differs from roundedQ32 product>0. |
| `npbench_weather_stencils_vadv_vadv__vadv` | Local constraint sharing; domain audit pending | One s=8h+r decomposition reconstructs source /4 and /8 rounded coefficients exactly; does not replace tridiagonal solve. Reciprocal divisor bounds still need comparison. |
| `pythran_allpairs__sqr_dists` | Reduction owner repair/audit | Materialized elementwise products/squares versus one fused rescale; allpairs identity and pairwise symmetry additionally require source algorithm restoration. |
| `pythran_allpairs__sqr_dists_loops` | Reduction owner repair/audit | Materialized elementwise products/squares versus one fused rescale; allpairs identity and pairwise symmetry additionally require source algorithm restoration. |
| `pythran_arc_distance__arc_distance` | Canonical gadget owner | Source high-level computation retained; exp/log/tanh/sin/cos/arctan2 approximation and rounding must switch to canonical APIs. |
| `pythran_bubble_sort__bubble_sort` | Restored here | Source adjacent bubble passes; break unreachable because source sets swap_test unconditionally. |
| `pythran_clip__clip` | Complex primitive audit pending | Raw isqrt(re²+im²) used for abs; need confirm canonical complex-abs rounding and dead-branch divisor substitution. |
| `pythran_create_grid__create_grid` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_cronbach__cronbach` | Reduction owner repair/audit | Materialized elementwise products/squares versus one fused rescale; allpairs identity and pairwise symmetry additionally require source algorithm restoration. |
| `pythran_crotg__CROTG` | Complex primitive audit pending | Raw isqrt magnitude and re²+im²==0 branch; compare canonical abs rounding and zero-domain behavior. |
| `pythran_d2q9_nxnyns__one_time_step` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_deuxd_convolution__conv` | Root convolution audit | Scalar/array product accumulation rounding and repeated clamped computations must match source/IR contract. |
| `pythran_euclidean_distance_square__euclidean_distance_square` | Reduction owner repair/audit | Materialized elementwise products/squares versus one fused rescale; allpairs identity and pairwise symmetry additionally require source algorithm restoration. |
| `pythran_fbcorr__fbcorr` | Reduction owner repair/audit | Materialized elementwise products/squares versus one fused rescale; allpairs identity and pairwise symmetry additionally require source algorithm restoration. |
| `pythran_fibo_seq__fibo` | Restored here | 1024 conditional sequential steps; extra state bounds removed to match field arithmetic. Intended int64 result scope n<=90 is not enforced by frozen compiler. |
| `pythran_goodexpomeasure__goodExpoMeasure` | Restored here; canonical dependency | Round-to-hundredths materialization, multiply255/truncate/wrap; exact rational shortcut removed. CanonicalQ32 may differ from NumPy bytes. |
| `pythran_harris__harris` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_hasting__fweb` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_insertion_sort__insertion_sort` | Restored here | Source insertion comparisons/shifts; no network or hinted permutation. |
| `pythran_ising__ising_step` | Restored here; canonical dependency | Literal exp comparison, not ln2 threshold; Q32 float32 contract. |
| `pythran_l2norm__l2_norm` | Reduction owner repair/audit | Materialized elementwise products/squares versus one fused rescale; allpairs identity and pairwise symmetry additionally require source algorithm restoration. |
| `pythran_lap__laplacian` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_lapl3d__laplacien` | Primitive choice allowed; domain audit pending | Stencil retained; max(1,max(out)) combined max gadget and denominator-specific quotient constraints, same formula. |
| `pythran_ln__ln` | Power primitive audit pending | Shared square-and-multiply powers2..9; check exact canonical PowFI multiplication schedule, not merely real polynomial equality. |
| `pythran_log_likelihood__log_likelihood` | Restored here; canonical dependency | Materialized per-element exp/normalize/log, not log identity. |
| `pythran_lu__factorMatrix0` | No algorithm replacement observed | Strict first-max pivot chain, entire-row swap, then elementwise elimination. abs/division edge domains pending. |
| `pythran_lu__factorMatrix1` | No algorithm replacement observed | Strict first-max pivot chain, suffix-row swap only, then elementwise elimination. Preserves distinction from factorMatrix0. |
| `pythran_make_decision__md` | Root repair | Raw squared distance removes rounded abs/square; affects ties and accepted domain. |
| `pythran_matmul__matrix_multiply` | Reduction owner repair/audit | Materialized elementwise products/squares versus one fused rescale; allpairs identity and pairwise symmetry additionally require source algorithm restoration. |
| `pythran_multi_export__a` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_multitype__check` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_multitype__check2` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_multitype__times` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_pairwise__pairwise` | Reduction owner repair/audit | Materialized elementwise products/squares versus one fused rescale; allpairs identity and pairwise symmetry additionally require source algorithm restoration. |
| `pythran_periodic_bc__periodic_bc` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_periodic_dist__dist` | Restored here | Every directed difference, both sequential masks, each directed square. Algebraic reverse-pair construction removed. |
| `pythran_perm__permutations` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_projection_simplex__projection_simplex` | Primitive choice allowed; domain audit pending | Source np.sort primitive; network/hinted sort allowed. Quotient reuse exact; onehot last-true lookup replaces compact+last gather, not a new projection algorithm. |
| `pythran_pythagorean_triples__next_pythagorean_triples` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_reverse_cumsum__reverse_cumsum` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_rosen__rosen` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_smoothing__smoothing` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_specialconvolve__specialconvolve` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_spectraldns__add_pressure_diffusion` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_spectraldns__cross2` | No replacement identified in scan | Source control/dataflow appears retained; shared arithmetic, intermediate bounds and accepted-domain equivalence are not yet certified. |
| `pythran_sumarray3d__summation` | Reduction owner repair/audit | Materialized elementwise products/squares versus one fused rescale; allpairs identity and pairwise symmetry additionally require source algorithm restoration. |
| `pythran_tax__tax` | Root repair | Four masked arrays replaced by one segment-selected polynomial; source structure needs restoration. |
| `pythran_value_index__tri` | Primitive choice allowed; domain audit pending | Source argsort primitive; stable index-key tie contract agrees with compiler, although not all NumPy unstable orders. |
| `pythran_vibr_energy__calculate_vibr_energy` | Canonical gadget owner | Source high-level computation retained; exp/log/tanh/sin/cos/arctan2 approximation and rounding must switch to canonical APIs. |
| `pythran_wdist__slow_wdist` | Reduction owner repair/audit | Materialized elementwise products/squares versus one fused rescale; allpairs identity and pairwise symmetry additionally require source algorithm restoration. |
| `pythran_weights__weights` | Local commutative CSE candidate | Equality-count pairs symmetric; same per-row threshold/count/reciprocal. Narrow quotient/threshold accepted domains require audit. |

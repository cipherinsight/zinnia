# CirC batch2 acceptance study

55 programs; 17 accepted with both fixture checks passing.

Each compilable translation was run locally using the supplied stock CirC executable, `--language zsharp ... r1cs --action count`, `timeout 300`, and the supplied `memrun.sh` with `MEMRUN_MAX=8G`. `TMPDIR` stayed inside this batch. The installed Z# stdlib was selected with `ZSHARP_STDLIB_PATH`. No Zinnia IR, circuits, sizes, or acceptance results were used to select a translation or category.

Real inputs use Q32 (scale 2^32), with complex values represented by real/imaginary pairs. All inputs remain private. Arrays preserve their declared dimensions; padded dynamic outputs include their logical dimensions or a presence tag. Accepted output values were decoded and compared against the original Python function, with decorators and annotations removed, using NumPy and `atol=rtol=1e-6`. These are A/B differential checks, not a proof over all inputs. Fixed-point quantization is mandated by the brief; IEEE nonfinite behavior is not represented.

The native `integer` type passes the Z# frontend but is unsupported in the R1CS backend. Arithmetic translations using it are retained with actual rejection logs. `probes/integer_field.log` also establishes that wrapping native integer arithmetic in field conversions does not fix the backend failure. Replacing these operations with finite-field division would change fixed-point semantics; custom rescaling/division gadgets were therefore not written. Missing sort/transcendental/search gadgets and truly unbounded shapes/loops are recorded as translation blockers, without compiling misleading stubs.

Data-movement-only programs can have zero constraints after CirC optimization. The 20,000-element reshape uses a statically expanded array literal: the equivalent repeated-assignment version compiled, but exhausted the interpreter memory cap. The final literal version compiles and both full-input zxi checks pass.

Reproduce all entries with `PYTHONDONTWRITEBYTECODE=1 /home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python run_batch.py`. The generator, original source copies, compile commands/logs, fixture `.pin` files, interpreter logs, expected outputs, decoded outputs, and per-program JSON records are retained here.

| Program | Category | Constraints | Reason |
|---|---|---:|---|
| case0296 | not_expressible: dynamic shape | — | kernel.py:11: width a.max()+1 has no bound in Integer[50]; fixture maxima are not domain bounds. |
| case0302 | accepted_with_rewrite | 62 | kernel.py:11: reshape has private dimensions; every valid reshape has exactly 20 elements, with dimensions divisors of 20. |
| case0312 | accepted_as_written | 107387 | Argmax scan uses strict comparison to preserve first tie, followed by static-shape unravel. |
| case0320 | accepted_as_written | 66257 | Argmin scan preserves first tie; unravel uses the declared 5x6 shape. |
| case0326 | needs_gadget: asin | — | kernel.py:11: requires asin; no matching fixed-point intrinsic in installed Z# stdlib. The brief forbids hand-writing this gadget. |
| case0332 | needs_gadget: gcd | — | kernel.py:11: np.gcd needs Euclidean iteration; unrestricted Integer signature gives no fixed iteration bound. No gcd intrinsic in installed stdlib. |
| case0339 | accepted_as_written | 0 | Static slicing, diagonal extraction, fliplr, and vstack. |
| case0346 | needs_gadget: fixed-point multiplication/rescaling | — | Private row mutation and cumsum preserved; native integer multiplication and Q32 rescale attempted. |
| case0355 | needs_gadget: sort | — | kernel.py:11: requires sort; no matching fixed-point intrinsic in installed Z# stdlib. The brief forbids hand-writing this gadget. |
| case0361 | accepted_as_written | 0 | Deleted columns [0,2] are source constants; output is statically 6x4. |
| case0368 | accepted_as_written | 24335 | isclose with atol=0 and rtol=1e-5: exact rational constant comparison in Q32 units, then all. |
| case0381 | needs_gadget: sort | — | kernel.py:11: requires sort; no matching fixed-point intrinsic in installed Z# stdlib. The brief forbids hand-writing this gadget. |
| case0387 | accepted_as_written | 0 | Static reshape/swapaxes/reshape mapped in C order; all 20,000 output elements retained. Static loop expansion uses an array literal to avoid zxi retaining successive full-array assignments; no value-dependent bounds changed. |
| case0393 | accepted_with_rewrite | 14915 | kernel.py:12: private slice width is at most 10 from input shape, including negative/clipped endpoints. |
| case0403 | accepted_as_written | 4 | C-order ravel_multi_index recurrence; a is retained as an unused private input, as in source. |
| case0411 | not_expressible: dynamic shape | — | kernel.py:26: both crop dimensions high_index-low_index are unbounded integers; 5x5 image does not bound padded crop size. |
| case0417 | needs_gadget: division | — | kernel.py:12: private reshape/binning/output length; 8 input items bound the number of nonempty bins. Division uses native integer Q32 units. |
| case0426 | not_expressible: dynamic shape | — | kernel.py:11: np.arange(m) and the 20 by m output have no bound on m. |
| case0433 | accepted_as_written | 1554 | Private dynamic row/column writes preserved, including valid NumPy negative indices. |
| case0439 | needs_gadget: fixed-point multiplication/rescaling | — | Broadcast outer products; all 20x10x10 outputs retained with Q32 integer arithmetic. |
| case0448 | accepted_as_written | 0 | Static transpose/vstack/transpose retaining both components. |
| case0454 | needs_gadget: division | — | Max absolute row norm then native integer division producing Q32 output. No hand-written fixed-point division gadget. |
| case0467 | accepted_with_rewrite | 76134 | kernel.py:22: private slice sizes make a ragged list; each slice has at most 4x4 elements from the declared source shape. |
| case0480 | accepted_as_written | 93 | Boolean argmax scan then original private-index verification and -1 fallback. |
| case0490 | accepted_with_rewrite | 674810 | kernel.py:13-15: np.where produces dynamic index vectors; each has capacity 50 from the row shape. |
| case0499 | not_expressible: dynamic shape | — | kernel.py:15-16,20: requested padding shape consists of two unbounded integer inputs. |
| case0505 | accepted_as_written | 114077 | Private take_along_axis index retained, then both sums; handles negative indices. |
| npbench_azimint_naive_azimint_naive | not_expressible: dynamic shape | — | kernel.py:12: output length npt is unbounded; npt != 0 gives no capacity bound. Mean also needs fixed-point division. |
| npbench_deep_learning_conv2d_bias_conv2d | needs_gadget: fixed-point multiplication/rescaling | — | Static convolution then bias, using native integer Q32 multiply/rescale. Source float32 storage represented in the brief-mandated fixed point. |
| npbench_mandelbrot1_mandelbrot1 | not_expressible: dynamic shape | — | kernel.py:10-14: xn and yn are unbounded input dimensions, and maxiter has no domain bound. |
| npbench_polybench_cholesky2_cholesky2 | needs_gadget: sqrt | — | kernel.py:10: requires sqrt; no matching fixed-point intrinsic in installed Z# stdlib. The brief forbids hand-writing this gadget. |
| npbench_polybench_doitgen_doitgen | needs_gadget: fixed-point multiplication/rescaling | — | kernel.py:10: reshape dimensions are private. The fixed C4 forces NP=16 and valid assignment forces 8x8x16; output exposes mutated A. |
| npbench_polybench_gesummv_gesummv | needs_gadget: fixed-point multiplication/rescaling | — | Preserve Python left associativity: (alpha*A)@x + (beta*B)@x. |
| npbench_polybench_k3mm_k3mm | needs_gadget: fixed-point multiplication/rescaling | — | Three matrix products in source order; Q32 integer rescaling at multiplication. |
| npbench_polybench_symm_symm | needs_gadget: fixed-point multiplication/rescaling | — | Expose mutated C as public output (source returns None); keep statically unrolled triangular slices. |
| npbench_scattering_self_energies_scattering_self_energies | needs_gadget: fixed-point multiplication/rescaling | — | All eight source loops, complex matrix products, private neighbor reads, and E>=w mask retained. Mutated Sigma is the output. |
| pythran_allpairs | needs_gadget: fixed-point multiplication/rescaling | — | Squared distances, outer Y then inner X order, with Q32 rescale. |
| pythran_babylonian | not_expressible: dynamic loop | — | kernel.py:12: Newton while loop and seen set grow with an unrestricted Integer input; no finite constant iteration bound follows from this signature. |
| pythran_cdotc | needs_gadget: fixed-point multiplication/rescaling | — | kernel.py:15,24: private trip count N; @requires 0<=N<=1024 supplies a bound, including zero strides. Complex multiply needs Q32 rescaling. |
| pythran_collatz_modified | not_expressible: dynamic loop | — | kernel.py:10,13: both while True loops have no syntactic or declared-domain bound; cannot invent a Collatz stopping bound. |
| pythran_d2q9_nxnyns | needs_gadget: fixed-point multiplication/rescaling | — | Source periodic boundaries, directional gathers, f2m/relaxation/m2f formulas and transposed setf are retained; Q32 native integer arithmetic. |
| pythran_euler_challenge14 | not_expressible: dynamic loop | — | kernel.py:29: MAX_NUM has no declared bound; recursive Collatz series_length at line 20 also lacks a bound. |
| pythran_fibo | circ_error | — | kernel.py:12-13,23-24: input-dependent recursion; @requires n<=1024 permits bounded recurrence. Preserve arbitrary integer result (F1024 exceeds a field). |
| pythran_guerre | not_expressible: dynamic loop | — | kernel.py:24: nter is unrestricted, so the convergence loop has no domain-implied constant bound. Complex division and magnitude also require gadgets. |
| pythran_ising | needs_gadget: exp | — | kernel.py:42: requires exp; no matching fixed-point intrinsic in installed Z# stdlib. The brief forbids hand-writing this gadget. |
| pythran_ldpc_decoder | needs_gadget: log | — | kernel.py:16: requires log; no matching fixed-point intrinsic in installed Z# stdlib. The brief forbids hand-writing this gadget. |
| pythran_make_circle_two_points | needs_gadget: sqrt | — | kernel.py:44: requires sqrt; no matching fixed-point intrinsic in installed Z# stdlib. The brief forbids hand-writing this gadget. |
| pythran_morphology | timeout_or_oom | — | kernel.py:18,29: private loop endpoints; image dimensions imply bounds 8 and 16, and requires bounds k to 0..1024. |
| pythran_nd_local_maxima | timeout_or_oom | — | Default mode=wrap; all static np.ndindex positions and offsets expanded, retaining source shape and all comparisons. |
| pythran_pivot | accepted_with_rewrite | 18113 | kernel.py:11-13,22: magnitude comparisons and private loop ranges. Any executed access j-1 into a length-two dimension must be in [-2,1]; nonempty valid iterations therefore have j in [-1,2]. |
| pythran_queens_numba | needs_gadget: search | — | kernel.py:21-33: recursive backtracking search with append/pop. Eight board rows bound live queens to 8; a padded stack still requires a search gadget, forbidden by brief. |
| pythran_shallow_water | not_expressible: dynamic shape | — | kernel.py:10,49-51: H and W are unrestricted input dimensions; I also supplies an unbounded iteration count. |
| pythran_spectraldns | needs_gadget: fixed-point multiplication/rescaling | — | Complex values encoded as real/imaginary pairs; preserve unusual source first cross component exactly. |
| pythran_train_eq | not_expressible: dynamic shape | — | kernel.py:13-14: TrSyms has no upper bound; os=0 permits arbitrarily many valid equal-sized windows, so E.shape does not bound it. |
| pythran_zero | accepted_with_rewrite | 1048576 | kernel.py:11: nested list shape is private; @requires explicitly bounds both dimensions by 1024. |

## Counts

| Category | Programs |
|---|---:|
| accepted_as_written | 11 |
| accepted_with_rewrite | 6 |
| circ_error | 1 |
| needs_gadget: asin | 1 |
| needs_gadget: division | 2 |
| needs_gadget: exp | 1 |
| needs_gadget: fixed-point multiplication/rescaling | 12 |
| needs_gadget: gcd | 1 |
| needs_gadget: log | 1 |
| needs_gadget: search | 1 |
| needs_gadget: sort | 2 |
| needs_gadget: sqrt | 2 |
| not_expressible: dynamic loop | 4 |
| not_expressible: dynamic shape | 8 |
| timeout_or_oom | 2 |

## Verification limits

All entries counted as accepted pass both A and B. Programs blocked at translation, rejected by CirC, or stopped by the resource cap have no verified circuit output and are not counted as accepted. Resource failures and errors are preserved in result.json and logs. No fixture-specific bound was used.

BATCH_DONE

# CirC acceptance study — batch1

All 55 programs have a `result.json`. **19/55 accepted (34.5%)**: 11 as written and 8 with minimal rewrites. Every accepted program passed fixtures A and B after decoding.

## Method and reproducibility

Only original kernel source, signatures, requirements, and fixtures were used for translation decisions. No Zinnia IR, circuits, sizes, or acceptance outcomes were used. Source and fixtures are copied into each program directory. Multi-function fixtures use `fixture.<function>.json` from the source directory.

All 32 executable translations were actually compiled with the supplied CirC binary using `--language zsharp <file> r1cs --action count`, wrapped by `memrun.sh`, `MEMRUN_MAX=8G`, and `timeout 300`. Each invocation has a `.command.json` and `.log`. `TMPDIR` is local to this batch. Binary hashes and numeric configuration are in `environment.json`. The remaining 23 programs have source-level non-expressibility or required-gadget assessments, not invented compiler failures; no executable stub was compiled for them.

Reals use integer fixed point with scale 2^32. Simple copying, addition, selection, and integer polynomial arithmetic use field values with centered signed decoding. Signed comparisons explicitly handle the centered representation. Native Z# `integer` was tried for signed wide fixed-point multiplication/rescaling; the R1CS backend rejected it. Those eight cases are classified as requiring a fixed-point multiplication/rescaling gadget, which the brief forbids writing. They are not counted as accepted merely because the frontend or interpreter supports integers. Field division was not substituted for fixed-point quotient: the reshape metadata division is exact on valid reshape inputs.

All input parameters remain private, including scalar shape parameters even where fixture notes call them constants. Returns are public. For procedures returning None, modified arrays are exposed and checked (CCOPY: CY; insertion sort: list2). Unsuccessful GEMVER/K2MM/spectralDNS translations likewise retain their observable modified arrays. Complex values are real/imaginary pairs. Padded outputs include logical length or dimensions; the decoder removes only padding and verifies reshape dimensions.

Run from this directory with `/home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python run.py`, then the same Python with `check.py`, then `report.py`. The checked-in `.zok` files and `manifest.json` are the authoritative final translations; `build.py`, `build_more.py`, and `normalize.py` are development generators, not the replay path. `run.py <program>...` and `check.py <program>...` support individual replay.

## Counts

| Category | Programs |
|---|---:|
| `accepted_as_written` | 11 |
| `accepted_with_rewrite` | 8 |
| `needs_gadget: division` | 3 |
| `needs_gadget: exp` | 1 |
| `needs_gadget: fixed-point multiplication` | 8 |
| `needs_gadget: sin` | 3 |
| `needs_gadget: sort` | 2 |
| `needs_gadget: sqrt` | 4 |
| `needs_gadget: tanh` | 1 |
| `not_expressible: dynamic loop` | 3 |
| `not_expressible: dynamic shape` | 6 |
| `timeout_or_oom` | 4 |
| `unsupported_feature: integer bitwise` | 1 |
| **Total** | **55** |

## Per-program results

Constraint counts are for the accepted attempt. A dash means no accepted R1CS; full diagnostics and rewrites are in the corresponding `result.json`.

| Program | Category | Constraints | Reason |
|---|---|---:|---|
| [case0295](case0295/result.json) | `not_expressible: dynamic shape` | — | The second zeros dimension is a.max()+1; Integer[50] gives no upper bound on values, so no finite padding capacity is implied. |
| [case0301](case0301/result.json) | `accepted_with_rewrite` | 4 | Twenty values plus public row/column metadata preserve the dynamic reshape. |
| [case0311](case0311/result.json) | `accepted_as_written` | 107546 | Scan for first argmax; unravel its flat index in Fortran order. |
| [case0319](case0319/result.json) | `accepted_as_written` | 8441 | Native private-index scatter and gather preserve the permutation. |
| [case0325](case0325/result.json) | `needs_gadget: sin` | — | np.sin(np.deg2rad(number)) and np.sin(number) need fixed-point sine; no supplied Z# implementation. |
| [case0331](case0331/result.json) | `not_expressible: dynamic loop` | — | np.gcd expands to a Euclidean remainder loop; unconstrained Integer operands give no declared iteration bound. |
| [case0338](case0338/result.json) | `accepted_as_written` | 0 | Static diagonal extraction and stacking. |
| [case0345](case0345/result.json) | `needs_gadget: fixed-point multiplication` | — | Native signed fixed-point arithmetic reaches R1CS, then crashes; a wide signed rescaling gadget is forbidden. |
| [case0354](case0354/result.json) | `accepted_with_rewrite` | 25810 | Pad symmetric row difference to nine rows and expose logical length. |
| [case0360](case0360/result.json) | `accepted_as_written` | 0 | Static deletion of row 2. |
| [case0366](case0366/result.json) | `needs_gadget: sort` | — | np.insert with an arbitrary position vector uses stable sorting of the adjusted insertion positions; a custom sorting gadget is prohibited. Output capacity itself is static (105 rows). |
| [case0375](case0375/result.json) | `needs_gadget: sort` | — | np.sort(x) requires a sorting implementation; no supplied Z# sort routine, and handwritten sorting networks are forbidden. |
| [case0386](case0386/result.json) | `accepted_as_written` | 58804 | Static expansion of all 19,701 windows; 78,804 public values. |
| [case0392](case0392/result.json) | `accepted_with_rewrite` | 20213 | Normalize private slice endpoints; pad to twenty rows with length. |
| [case0402](case0402/result.json) | `accepted_as_written` | 7 | Fortran-order ravel via private dimension products. |
| [case0409](case0409/result.json) | `accepted_as_written` | 0 | Elementwise fixed-point addition. |
| [case0416](case0416/result.json) | `needs_gadget: division` | — | The dynamic reshape can be padded to 8x6 (six columns bound the number of bins), but mean(axis=-1) then needs signed fixed-point division by bin_size. |
| [case0425](case0425/result.json) | `not_expressible: dynamic shape` | — | np.arange(m) and the 20xm output depend on unconstrained Integer m; no finite output capacity is implied. |
| [case0431](case0431/result.json) | `needs_gadget: sqrt` | — | np.std computes a square root of a data-dependent variance; fixed-point sqrt is not supplied. |
| [case0438](case0438/result.json) | `needs_gadget: sqrt` | — | np.corrcoef normalizes covariance by data-dependent standard deviations, requiring sqrt and fixed-point division. |
| [case0444](case0444/result.json) | `accepted_with_rewrite` | 101246 | Pad Boolean selection to twenty entries and expose length. |
| [case0453](case0453/result.json) | `needs_gadget: sqrt` | — | Data-dependent L2 norms need square root and fixed-point normalization division. |
| [case0463](case0463/result.json) | `accepted_with_rewrite` | 5169 | Pad selected nonzero run starts to twenty entries and expose length. |
| [case0478](case0478/result.json) | `accepted_as_written` | 12 | Preserve diagonal-view aliasing when zeroing the array. |
| [case0489](case0489/result.json) | `timeout_or_oom` | — | Masked 100x50 rewrite killed under the 8 GB cap. |
| [case0498](case0498/result.json) | `not_expressible: dynamic shape` | — | np.pad output dimensions come from unconstrained Integer[2] shape; source dimensions do not bound padding above. |
| [case0504](case0504/result.json) | `accepted_as_written` | 130976 | Native private gather along the last axis, followed by sum. |
| [npbench_azimint_hist_azimint_hist](npbench_azimint_hist_azimint_hist/result.json) | `not_expressible: dynamic shape` | — | Both histograms have npt bins; unconstrained Integer npt has no upper bound, including when there are only 16 samples. |
| [npbench_crc16_crc16](npbench_crc16_crc16/result.json) | `unsupported_feature: integer bitwise` | — | Native Integer bitwise rejected; unsigned narrowing was discarded. |
| [npbench_go_fast_go_fast](npbench_go_fast_go_fast/result.json) | `needs_gadget: tanh` | — | np.tanh on private diagonal elements requires a fixed-point transcendental routine. |
| [npbench_polybench_bicg_bicg](npbench_polybench_bicg_bicg/result.json) | `needs_gadget: fixed-point multiplication` | — | Native signed fixed-point arithmetic reaches R1CS, then crashes; a wide signed rescaling gadget is forbidden. |
| [npbench_polybench_deriche_deriche](npbench_polybench_deriche_deriche/result.json) | `needs_gadget: exp` | — | Private alpha is used in np.exp and real exponentiation, plus fixed-point division. |
| [npbench_polybench_gemver_gemver](npbench_polybench_gemver_gemver/result.json) | `needs_gadget: fixed-point multiplication` | — | Native signed fixed-point arithmetic reaches R1CS, then crashes; a wide signed rescaling gadget is forbidden. |
| [npbench_polybench_k2mm_k2mm](npbench_polybench_k2mm_k2mm/result.json) | `needs_gadget: fixed-point multiplication` | — | Native signed fixed-point arithmetic reaches R1CS, then crashes; a wide signed rescaling gadget is forbidden. |
| [npbench_polybench_seidel_2d_seidel_2d](npbench_polybench_seidel_2d_seidel_2d/result.json) | `not_expressible: dynamic loop` | — | The outer loop runs TSTEPS-1 times; only N is bounded by requires, not TSTEPS. |
| [npbench_pythran_arc_distance_arc_distance](npbench_pythran_arc_distance_arc_distance/result.json) | `needs_gadget: sin` | — | Private angle arrays need sin, cos, sqrt, and arctan2; no corresponding fixed-point standard-library routines. |
| [pythran_allpairs](pythran_allpairs/result.json) | `needs_gadget: fixed-point multiplication` | — | Native signed fixed-point arithmetic reaches R1CS, then crashes; a wide signed rescaling gadget is forbidden. |
| [pythran_arc_distance](pythran_arc_distance/result.json) | `needs_gadget: sin` | — | Private angle arrays need sin, cos, sqrt, and arctan2; no corresponding fixed-point standard-library routines. |
| [pythran_ccopy](pythran_ccopy/result.json) | `accepted_with_rewrite` | 1854207 | Guard 1,024 copy iterations from declared N bound; return mutated CY. |
| [pythran_collatz](pythran_collatz/result.json) | `timeout_or_oom` | — | 1022 slots and 178 proven rounds; killed under the 8 GB cap. |
| [pythran_crotg](pythran_crotg/result.json) | `needs_gadget: sqrt` | — | Complex magnitude and NORM require fixed-point sqrt, followed by division. |
| [pythran_euclidean_distance_square](pythran_euclidean_distance_square/result.json) | `needs_gadget: fixed-point multiplication` | — | Native signed fixed-point arithmetic reaches R1CS, then crashes; a wide signed rescaling gadget is forbidden. |
| [pythran_fft](pythran_fft/result.json) | `needs_gadget: fixed-point multiplication` | — | Native signed fixed-point arithmetic reaches R1CS, then crashes; a wide signed rescaling gadget is forbidden. |
| [pythran_growcut](pythran_growcut/result.json) | `timeout_or_oom` | — | 1024x1024 capacity and masked neighborhoods; killed under the 8 GB cap. |
| [pythran_insertion_sort](pythran_insertion_sort/result.json) | `accepted_with_rewrite` | 374917 | Bound source insertion-sort while by nine guarded shifts. |
| [pythran_laplace](pythran_laplace/result.json) | `timeout_or_oom` | — | 1024x1024 grid and 1024 time steps; killed under the 8 GB cap. |
| [pythran_lu](pythran_lu/result.json) | `needs_gadget: division` | — | M[j,line] / pivot needs signed fixed-point division on private data. |
| [pythran_monte_carlo](pythran_monte_carlo/result.json) | `not_expressible: dynamic loop` | — | numSteps is constrained only to be nonzero; its range loops have no finite declared upper bound. Sine and division are additional obstacles. |
| [pythran_multitype](pythran_multitype/result.json) | `accepted_as_written` | 1023 | Multiplication with declared 0..1,024 scalar domains. |
| [pythran_perm](pythran_perm/result.json) | `accepted_with_rewrite` | 92 | Preserve cycles algorithm; exactly 4! outputs from a four-element input. |
| [pythran_pythagorean_triples](pythran_pythagorean_triples/result.json) | `accepted_as_written` | 6 | Static integer matrix multiplication, transpose and reshape. |
| [pythran_sexy_primes](pythran_sexy_primes/result.json) | `not_expressible: dynamic shape` | — | The primes output and range(9,x+1) depend on unconstrained Integer x; no finite capacity follows from the signature. |
| [pythran_spectraldns](pythran_spectraldns/result.json) | `needs_gadget: fixed-point multiplication` | — | Native signed fixed-point arithmetic reaches R1CS, then crashes; a wide signed rescaling gadget is forbidden. |
| [pythran_train_eq](pythran_train_eq/result.json) | `not_expressible: dynamic shape` | — | err has TrSyms elements with no bound on TrSyms; os may be zero, so the fixed E shape cannot bound the number of iterations. |
| [pythran_weights](pythran_weights/result.json) | `needs_gadget: division` | — | weights[i] = 1 / total needs fixed-point reciprocal of a private match count; handwritten division or lookup gadgets are forbidden. |

## Limits and verification gaps

- A/B checks establish fixture correctness, not an exhaustive proof over every numeric input; fixed-point quantization and the finite field representation remain numerical limits of the translation. No observed fixture range was used to invent array or loop bounds.
- The 36 programs without an accepted attempt have no completed A/B equivalence check. Their classifications distinguish source-level impossibility/gadget requirements from actual compilation failure. Four capped runs ended with signal 9: masked update, Collatz, GrowCut, and Laplace; the scope memory cap was 8 GB. None is counted as accepted.
- Collatz has a domain-derived bound: exhaustive integer simulation of every initial value 3..1024 gives at most 178 transitions and maximum intermediate value 250504. The proof data are in `pythran_collatz/bound.json`; this does not use fixture values.
- The large sliding-window interpreter initially exceeded the cap with repeated output-array mutation. Expanding the same static window expressions into an array literal preserved every output and allowed both fixtures to complete; the final source, compilation log, and checks correspond to that representation.
- CRC16 was not credited for an exploratory u64 translation: although it passed A/B, unconstrained signed Integer poly was narrowed. The final native-Integer attempt reports the actual bitwise-operand rejection. The discarded investigation is explicitly separated under `npbench_crc16_crc16/discarded_u64/`.
- Fixed-point division, transcendental functions, and library sorting were not replaced with handwritten gadgets. FFT twiddles were evaluated only as compile-time constants; its private fixed-point arithmetic still failed in R1CS. GrowCut’s zero color distance was folded only because image is initialized to zero and never modified for any input.

BATCH_DONE

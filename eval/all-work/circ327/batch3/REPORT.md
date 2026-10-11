**Batch3: CirC acceptance study — 54 programs**

17/54 programs accepted and passed both original fixtures: 9 as written and 8 with minimal rewrites. No accepted result has an unchecked or mismatching A/B output.

All materialized attempts were actually compiled with the supplied stock CirC binary, `--language zsharp ... r1cs --action count`, under `MEMRUN_MAX=8G memrun.sh timeout 300`. Exact argv, environment, return code, elapsed time and compiler output are next to each attempt. `environment.json` records binary hashes; `manifest.json` records source and fixture hashes. Source/fixture snapshots are retained per program. No Zinnia IR, circuits, sizes or acceptance outcomes were used in the classifications.

All signature parameters remain private, including parameters that fixture prose calls constants. Static NumPy dimensions remain unchanged in direct translations. Dynamic rewrites expose lengths/shapes alongside padding; each result records the domain argument and every rewrite. Programs with no bounded representation or requiring a prohibited gadget have documentary attempts, not placeholder programs or invented compiler results.

Reals are rounded to integers at scale 2^32 and decoded from centered field residues. Direct translations use native field operations; constant integer rescaling uses native remainder and division, not a hand-built division circuit. Private real division, sqrt, log, tanh and missing sorting primitives are recorded as needing gadgets. The native field remainder lowering crashes in this stock binary (`circ_fields/src/lib.rs:571:56`, `Option::unwrap()` on `None`); four static kernels were also tried with native arbitrary integers, which the R1CS backend rejects with `Unhandled variable operator return...`. Fibonacci retains arbitrary integers because its declared n<=1024 permits outputs wider than the default field.

Validation executes the original Python/NumPy function after removing only Zinnia imports, decorators and annotations. No Zinnia compilation is involved. Mutated arrays are public outputs for in-place kernels. Every accepted attempt has A.pin/B.pin, interpreter logs, decoded/reference values and maximum error in result.json, compared with relative and absolute tolerance 1e-6. Zero constraints on reshapes and wiring-only kernels are actual CirC counts. Complex struct output is decoded from zxi’s alphabetical (im,re) presentation.

| Category | Programs |
|---|---:|
| accepted_as_written | 9 |
| accepted_with_rewrite | 8 |
| circ_error | 10 |
| needs_gadget: division | 4 |
| needs_gadget: log | 1 |
| needs_gadget: sort | 3 |
| needs_gadget: sqrt | 6 |
| needs_gadget: tanh | 1 |
| not_expressible: dynamic loop | 2 |
| not_expressible: dynamic shape | 8 |
| timeout_or_oom | 2 |

| Program | Category | Constraints | Reason |
|---|---|---:|---|
| [case0297](case0297/result.json) | not_expressible: dynamic shape | — | kernel.py:12: zeros((50, temp.max()+1)); unrestricted Integer values make max(a)-min(a)+1 unbounded. |
| [case0303](case0303/result.json) | accepted_with_rewrite | 23719 | Pad at 23 elements; public reshape dimensions, private ncol retained. |
| [case0313](case0313/result.json) | accepted_as_written | 107388 | Native signed Q32 maximum scan; first argmax preserved. |
| [case0321](case0321/result.json) | accepted_as_written | 66257 | Native maximum scan returns row and column. |
| [case0327](case0327/result.json) | not_expressible: dynamic shape | — | kernel.py:11: np.pad output length is private unrestricted length; length >= 10 has no upper bound. |
| [case0334](case0334/result.json) | circ_error | — | Native field remainder crashes; alternate arbitrary-integer Q32 arithmetic also rejected. |
| [case0340](case0340/result.json) | accepted_as_written | 0 | Flatten the fixed 5x6 input in row-major order. |
| [case0347](case0347/result.json) | needs_gadget: division | — | kernel.py:11: a[row-1,:] /= divide_number requires division by a private Q32 real, not field inversion. |
| [case0356](case0356/result.json) | needs_gadget: sort | — | kernel.py:11: np.argsort(a,axis=0) needs a sorting/argsort implementation; there is no native Z# sort in the available stdlib. |
| [case0362](case0362/result.json) | accepted_with_rewrite | 60015 | Six-column keep mask, padded compact output and column count. |
| [case0369](case0369/result.json) | accepted_as_written | 19826 | Native tolerance comparison and all reduction. |
| [case0382](case0382/result.json) | needs_gadget: sort | — | kernel.py:11: np.argsort(a) requires sorting 100 private reals. No handwritten sorting network or permutation argument is permitted. |
| [case0388](case0388/result.json) | timeout_or_oom | — | 20,000-slot padded patch extraction exhausted the 8 GB memory cap. |
| [case0398](case0398/result.json) | accepted_as_written | 59 | Original 30-step linear recurrence, integer coefficients. |
| [case0405](case0405/result.json) | not_expressible: dynamic shape | — | kernel.py:11: bincount produces max(accmap)+1 entries; nonnegative accmap has no upper bound in Integer[100]. |
| [case0412](case0412/result.json) | accepted_with_rewrite | 7081 | Ten-slot masked compaction and public length. |
| [case0418](case0418/result.json) | circ_error | — | Six-bin padded rewrite reaches native field remainder backend crash. |
| [case0427](case0427/result.json) | not_expressible: dynamic shape | — | kernel.py:14: arange(m) and the output have private unrestricted width m; input count 20 does not bound m. |
| [case0434](case0434/result.json) | accepted_as_written | 2980 | Native dynamic row/column writes, including negative indices. |
| [case0440](case0440/result.json) | needs_gadget: sqrt | — | kernel.py:13: np.sqrt(np.diag(mat)) requires fixed-point square root. |
| [case0449](case0449/result.json) | accepted_as_written | 0 | Native transpose/stack indexing; 3x4x2 output retained. |
| [case0456](case0456/result.json) | needs_gadget: sqrt | — | kernel.py:11: Euclidean norm requires square root of each sum of squares. |
| [case0468](case0468/result.json) | accepted_with_rewrite | 78506 | Sixteen padded windows; radii saturate only where slices are identical. |
| [case0481](case0481/result.json) | accepted_with_rewrite | 615 | Nine-slot compaction of matching indices and public length. |
| [case0491](case0491/result.json) | accepted_as_written | 16770 | Native isclose comparisons and mismatch count. |
| [case0500](case0500/result.json) | accepted_as_written | 0 | Static 54-element to 18x3 reshape. |
| [case0507](case0507/result.json) | accepted_with_rewrite | 79198 | Guarded 5x6 bounding-box crop with public dimensions. |
| [npbench_cavity_flow_cavity_flow](npbench_cavity_flow_cavity_flow/result.json) | needs_gadget: division | — | kernel.py:33-34: private b shape and private nt/nit loops are not expressible as written. For nt>0, build_up_b assigns a 14x14 RHS into b[1:-1,1:-1], forcing ny=nx=16 on valid calls; for nt=0 b is unobserved. A guarded 16x16 allocation and two loops bounded by declared 1024 preserve all outputs. Lines 10,18,39-40 still require private fixed-point division. |
| [npbench_deep_learning_lenet_lenet](npbench_deep_learning_lenet_lenet/result.json) | circ_error | — | Full bounded-shape convolution/pooling/dense translation hits native remainder crash. |
| [npbench_mandelbrot2_mandelbrot2](npbench_mandelbrot2_mandelbrot2/result.json) | not_expressible: dynamic shape | — | kernel.py:11-18: mgrid and complex arrays have xn*yn elements; xn and yn have no bound. The itermax <= 1024 precondition only bounds iterations. |
| [npbench_polybench_cholesky_cholesky](npbench_polybench_cholesky_cholesky/result.json) | needs_gadget: sqrt | — | kernel.py:10,16: Cholesky requires private square roots; line 14 also needs fixed-point division. |
| [npbench_polybench_durbin_durbin](npbench_polybench_durbin_durbin/result.json) | needs_gadget: division | — | kernel.py:16: alpha divides by private real beta; modular field division is not fixed-point division. |
| [npbench_polybench_gramschmidt_gramschmidt](npbench_polybench_gramschmidt_gramschmidt/result.json) | needs_gadget: sqrt | — | kernel.py:14: sqrt(nrm); line 15 also divides by the private real R[k,k]. |
| [npbench_polybench_lu_lu](npbench_polybench_lu_lu/result.json) | needs_gadget: division | — | kernel.py:13: A[i,j] /= A[j,j] divides private fixed-point quantities. |
| [npbench_polybench_syr2k_syr2k](npbench_polybench_syr2k_syr2k/result.json) | circ_error | — | Native Q32 rescaling crashes; alternate arbitrary integers also rejected. |
| [npbench_spmv_spmv](npbench_spmv_spmv/result.json) | circ_error | — | Sixteen guarded sparse terms; native Q32 rescaling crashes. |
| [pythran_allpairs_distances](pythran_allpairs_distances/result.json) | not_expressible: dynamic shape | — | kernel.py:14-15: arange(600*d) and arange(200*d) have unbounded private sizes; d has no declared bound. No algebraic closed-form replacement is a minimal statement translation. |
| [pythran_blacksholes](pythran_blacksholes/result.json) | needs_gadget: sqrt | — | kernel.py:20 is a dynamic loop, bounded by nb_opt <=1024 (and 14 array entries on valid calls); lines 21-30 require private sqrt/log/exp/division. Pad two outputs to 14 entries and guard 14 iterations, but the numerical gadgets remain required. |
| [pythran_cdotu](pythran_cdotu/result.json) | circ_error | — | 1024 guarded complex multiply-add steps; native Q32 rescaling crashes. |
| [pythran_compute_subpix_2d_gaussian2](pythran_compute_subpix_2d_gaussian2/result.json) | needs_gadget: log | — | kernel.py:10-12: crop/where lists are dynamic but at most 4x4; masks and padding can bound them. Lines 21-26 call log on private crop entries; lines 28-29 additionally need fixed-point division. |
| [pythran_deriv](pythran_deriv/result.json) | circ_error | — | Valid-array bounds permit three steps; native Q32 rescaling crashes. |
| [pythran_factorize_naive](pythran_factorize_naive/result.json) | accepted_with_rewrite | 55078 | Declared n<=1024 implies 32 steps and ten factor slots; native u32 division. |
| [pythran_fibo_seq](pythran_fibo_seq/result.json) | circ_error | — | 1024 guarded steps preserve full integer result; R1CS rejects native integer output. |
| [pythran_harris](pythran_harris/result.json) | circ_error | — | Native Q32 multiplication rescaling crashes; alternate arbitrary integers also rejected. |
| [pythran_l2norm](pythran_l2norm/result.json) | needs_gadget: sqrt | — | kernel.py:10: np.sqrt of a sum of squares needs a fixed-point square-root gadget. |
| [pythran_ln](pythran_ln/result.json) | circ_error | — | Native constant rescaling crashes; alternate arbitrary integers also rejected. |
| [pythran_make_decision](pythran_make_decision/result.json) | accepted_with_rewrite | 172256 | Compare squared complex distances directly; retain original first argmin. |
| [pythran_mulmod](pythran_mulmod/result.json) | not_expressible: dynamic loop | — | kernel.py:11: while x > 0 has bit_length(x) iterations; only m is bounded by the precondition, not x. Assuming a 32/64-bit x would narrow the Integer domain. |
| [pythran_nqueens](pythran_nqueens/result.json) | not_expressible: dynamic shape | — | kernel.py:45: list(range(queen_count)) and solution tuples have unbounded width; queen_count has no bound. Permutations and solution list sizes are also unbounded. |
| [pythran_primes_sieve](pythran_primes_sieve/result.json) | not_expressible: dynamic shape | — | kernel.py:9: [True] * limit has unrestricted private length; no signature bound on limit. The returned prime list is also dynamic. |
| [pythran_quicksort](pythran_quicksort/result.json) | timeout_or_oom | — | Bounded original partition and explicit stack: timeout; interpreter independently matches A/B. |
| [pythran_slowparts](pythran_slowparts/result.json) | needs_gadget: tanh | — | kernel.py:12: tanh is required on private real array elements. Nonempty valid assignments at line 16 imply d=8,re=5 from the 9x5 RHS; d=0 skips the loop and has empty output with variable shape metadata. A 9x5x16x8 capacity plus shape metadata suffices, but tanh remains necessary. |
| [pythran_stone](pythran_stone/result.json) | not_expressible: dynamic loop | — | kernel.py:18,37: N2=12*loopstart and range(1,N2+1) have no declared upper bound. Replacing the full kernel with its final KIP expression would discard source statements; transcendental calls at lines 70-71,94 additionally need gadgets. |
| [pythran_value_index](pythran_value_index/result.json) | needs_gadget: sort | — | kernel.py:11: np.argsort(score[:,1]) is not a native Z# operation; implementing a new sorting gadget is prohibited. |

Limits of verification: A/B comparisons are fixture checks, not an exhaustive proof of equivalence or of absence of finite-field/fixed-point overflow on every mathematical real/integer. No fixture numeric range was used to invent a loop or allocation bound. Nonaccepted programs have no verified R1CS circuit; construct/gadget classifications are source-and-library assessments, whereas compiler errors and resource failures are measured. The original quicksort translation additionally passed both fixtures in zxi even though that alone does not establish R1CS acceptance. Factorization also passed interpreter checks at n=0,1,2,1021,1024 (edge_checks.json).

Reproduce retained attempts with `/home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python reproduce.py` from this directory, or append program names for a subset. Run `write_report.py` after reproducing to refresh this report. Reproduction uses the supplied source/fixture locations, whose hashes and local snapshots are recorded.

BATCH_DONE

> Coordinated final validation: **75/75 profile checks and 25/25 Halo2 proofs pass**, and all 25 task programs pass the actual study worker. The task review below preserves the baseline and repair history. See [the final audit report](../README.md) and [result manifest](../final-manifest.json) for the verified build and complete evidence.

# POOL adversarial review

25 ordinary Python/NumPy implementations are retained unchanged, including all failures. The task is non-overlapping 2×2 **sum** pooling of NHWC integer arrays, with positive B/C and even positive H/W. Local helpers are intentional ordinary Python, not out of scope.

## Baseline results

- 48/75 profile checks pass; 16/25 implementations pass all three profiles.
- Every implementation agrees with the independent NumPy oracle on 19 fixtures in each of three profiles (1,425 Python cases).
- Successful profile checks compile both the unrewritten source and instrumented source, execute all 19 fixture assertions in the mock circuit, and reject a deliberately incorrect expected result.
- The 48 successful profile checks cover 912 compiled fixture cases and 48 negative controls; failing add-out checks each pass the initial all-zero fixture before rejecting the next fixture (915 compiled cases total). Failures are not counted as success merely because the original source compiles.
- `baseline.json` contains exact errors, stages, timings, source hashes, and environment metadata. No shared compiler files were changed or builds performed by this reviewer.

## Inventory

| Candidate | Intent | Passing profiles |
|---|---|---|
| `scripts_four_slices.py` | Add the four strided pixel grids directly. | 0,1,2 |
| `scripts_loops_scalar.py` | Write each batch, output row, output column and channel using scalar loops. | 0,1,2 |
| `scripts_patch_sum.py` | Loop over spatial blocks and reduce each 2x2 patch over spatial axes. | 0,1,2 |
| `scripts_nested_list.py` | Build the result using nested list comprehensions and scalar sums. | 0,1,2 |
| `scripts_append_blocks.py` | Append block sums to ordinary Python lists before converting to an array. | 0,1,2 |
| `scripts_enumerate_batches.py` | Enumerate input batches and assign vector block sums. | none |
| `scripts_zip_rows.py` | Zip alternating row grids, then pair columns in each row. | 0,1,2 |
| `scripts_while_blocks.py` | Use while loops to traverse non-overlapping blocks. | 0,1,2 |
| `scripts_local_helper.py` | Factor the four-grid addition into a local helper. | none |
| `scripts_closure_helper.py` | Capture the input in a local block helper. | none |
| `scripts_reshape_tuple_sum.py` | Expose block axes through reshape and sum both axes together. | 0,1,2 |
| `scripts_reshape_sequential_sum.py` | Reduce horizontal then vertical pair axes separately. | 0,1,2 |
| `scripts_swapaxes_blocks.py` | Move the vertical pair axis beside the horizontal pair axis and flatten each block. | 0,1,2 |
| `scripts_channel_first.py` | Transpose to channel-first layout for pooling then restore NHWC. | 0,1,2 |
| `scripts_einsum.py` | Sum the two block axes using an explicit einsum contraction. | 0,1,2 |
| `scripts_ufunc_reduce.py` | Use NumPy add.reduce over the exposed block axes. | 0,1,2 |
| `scripts_reduceat.py` | Perform segmented addition every two rows and columns with add.reduceat. | none |
| `scripts_window_view.py` | Select every other sliding 2x2 window and sum the window axes. | none |
| `scripts_stack_slices.py` | Stack the four strided grids along a new axis before summing. | 0,1,2 |
| `scripts_python_sum_slices.py` | Use Python sum over a generator of the four strided grids. | 0,1,2 |
| `scripts_copy_inplace.py` | Copy the first pixel grid and accumulate the other three in place. | 0,1,2 |
| `scripts_add_out.py` | Accumulate paired rows and columns with the NumPy add out argument. | none |
| `scripts_split_pairs.py` | Split each batch into two-row and two-column blocks before reducing. | none |
| `scripts_ndindex.py` | Traverse output coordinates with np.ndindex and sum four scalar pixels. | none |
| `scripts_enumerate_add.py` | Enumerate batches and combine scalar-indexed channel vectors through explicit np.add calls. | none |

## Failure groups and repair guidance

1. **`scripts_add_out.py`: semantic mutation failure.** Both original and instrumented programs compile, but every profile rejects an expected answer during execution. NumPy passes all fixtures. Honor `np.add(..., out=...)` by writing results to the destination storage and preserving alias/view behavior; accepting syntax without mutation is unsound.

2. **`scripts_enumerate_batches.py` and `scripts_enumerate_add.py`: ndarray identity across iteration/indexing.** Assigning four summed C-vectors reports 4*C RHS elements against C target elements, even on the singleton profile. The explicit `np.add` variant reproduces the operator variant, ruling out a problem unique to Python `+` syntax. Suspected legacy List conversion/concatenation rather than elementwise addition; inspect enumerate dispatch, tuple unpacking, trailing-axis subscript results, and binary dispatch. Preserve array type through these paths. This is a diagnosis hypothesis, not a completed repair.

3. **`scripts_local_helper.py` and `scripts_closure_helper.py`: local function rejection.** The compiler rejects both with a nested-def error claiming captured variables. The first helper captures nothing; the second intentionally captures `x`. Safely lift local helper bodies with explicit captured-value arguments or support static closure environments. Requiring users to decorate/move ordinary local helpers would narrow the accepted algorithm rather than repair it.

4. **`scripts_ndindex.py`: missing `np.ndindex`.** The shape is statically known. Support ordinary row-major coordinate iteration over a shape tuple.

5. **`scripts_reduceat.py`: ufunc method dispatch.** `np.add.reduceat` fails with misleading `Missing required argument: x1`; `np.add.reduce` is already accepted. Add segmented axis reduction dispatch for static index arrays and preserve signed sum semantics.

6. **`scripts_split_pairs.py`: accepted original, unusable returned value.** Original compilation succeeds, but checked compilation hits `IRBuilder::ensure_ptr: cannot materialize None as a scalar IR pointer`. Relevant legacy path: `numpy_funcs.rs` np.split dispatch -> `static_ndarray_ops/reshaping.rs` split_impl/slice_along_axis. The latter operates on List/Tuple and otherwise clones input; array identity and nested split method reductions deserve inspection. Preserve split results as arrays supporting `.sum`, and ensure original return values are actually typed/validated.

7. **`scripts_window_view.py`: missing NumPy nested namespace/view.** Compilation rejects `np.lib`. The static sliding-window operation is a natural implementation of this task; support the namespace and shape/stride view, or lower the operation to explicit gathers. Do not prohibit window algorithms merely to increase pass rates.

## Scope and limits

The runner uses mock circuit execution, not a cryptographic proof backend. These 25 examples are a bounded adversarial search, not evidence that all NumPy programs compile. Fixtures include extreme signed values and multiple shapes as provided by the shared checker. Original candidates and baseline errors remain available for regression checks after root integration.


## API repair integration

The root integration now passes **69/75 profile checks (23/25 candidates)**, recorded in `post-api.json`. All original `scripts_*.py` candidate sources remain unchanged. Only the local-helper and closure-helper candidates still fail; root is integrating general helper lowering separately.

This reviewer implemented `src/ir_gen/named_attr/array_iteration.rs`, with root registering the dispatch and rebuilding. The module adds static-shape `np.ndindex`, minimum/maximum ufunc accumulation, add.reduceat, sliding_window_view, roll, and edge padding. It uses general shape/axis operations rather than POOL or SMOOTH pattern substitutions. Window results preserve the source segment, offset, and strides so reads reflect source mutations. Reductions and copied outputs retain StaticArray type; `out=` uses the shared write-through helper.

Root also repaired array iteration identity, split views, and output mutation. The original enumeration, split, and out= candidates now pass all profiles.

`_api_probes.py` / `api-probes.json` independently validate the added APIs: **32/32 probes pass**, comprising 22 valid programs with three signed-value fixtures each, original and instrumented compilation, and 22 wrong-answer rejections; plus 10 invalid/unsupported argument rejection probes. Covered cases include negative/repeated axes, descending/repeated reduceat boundaries, noncontiguous inputs, scan out-aliases, source mutations observed through windows, roll argument broadcasting, and asymmetric padding.

Explicit API limits: shape, axes, reduction indices, shifts, and pad widths must be compile-time known; runtime-shape DynamicNDArray is rejected. Accumulator dtype conversions, subok=True, and writeable=True are rejected; matching dtypes are supported, including int, np.int64, x.dtype, and the string "int64". The existing frontend collapses NumPy integer-width class aliases to Integer, so width distinctions are not available to this module; narrow dtype strings are explicitly rejected. Other pad modes fall through to existing implementations. The StaticArray type has no read-only flag, so this implementation supports read-only window usage and rejects explicitly requested writable windows, but does not enforce NumPy's default write protection against later assignments into the returned view.


## Final integrated verification

**All 25 unchanged candidates now pass all three profiles: 75/75 checks**, recorded in `../results/pool-after.json`. This covers 1,425 NumPy oracle cases, 1,425 compiled mock cases, both original and checked compilation, and 75 incorrect-answer negative controls. General helper lowering resolves both remaining local-helper variants without rewriting the retained candidate sources.

After adding matching explicit accumulator dtype support, **40/40 API probes pass**: 28 valid programs × three fixtures, 28 wrong-answer rejections, and 12 invalid/unsupported argument rejections. `api-probes.json` contains the final results. The six added valid probes exercise int, np.int64, x.dtype, and string dtype forms for scans and reduceat.


## Additional aliasing regressions

A shared regression found that split results and positive-step basic slices could lose source aliases. The repair now routes original array arguments into the split family and constructs views retaining the source segment, offset, and strides. `src/helpers/static_array/read.rs` now preserves compile-time positive-step basic slices centrally, including mixed scalar/slice indexing such as output columns. Negative-step slices retain the existing materializing path because StaticArray strides are unsigned.

Explicit split boundaries now follow NumPy for negative and descending indices, ndarray boundary vectors are supported, and nonpositive section counts are rejected before allocating sections.

The expanded API suite passes **51/51 probes**: 38 valid programs × three signed fixtures with 38 wrong-answer rejections, and 13 argument rejection checks. The four shared split/out regression checks also pass. New probes verify mutation through split/hsplit/dsplit views, overlap from descending split boundaries, strided slice aliases in both directions, ufunc outputs into strided slices, and reductions into output columns.

The first full Halo2 run passed **25/25 candidates** on profile 0, fixture 6, stored in `../results/pool-halo2.json`. A final compiler-build gate will refresh the complete task and proof artifacts after integration finishes.

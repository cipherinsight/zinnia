# Revision 2: composed numerical array tasks

New participant assignments use six versioned tasks. The original pilot task IDs,
submissions, ratings, times, examples, and grading contracts are retained.

The revision increases the number of interacting array operations rather than
merely increasing input sizes. Both conditions receive the same specification,
inputs, oracle, manual access, hints, and time allowance. Pure loops are explicitly
accepted in every hint; no preferred algorithm is required.

| Task | Required computation | Output | Provisional minutes |
|---|---|---|---:|
| Batched channel energy | Channel calibration and clipping, per-batch/channel temporal minimum, squared residual reduction | `[B,C]` | 10 |
| Centered channel products | Integer-scaled centering followed by all cross-channel products, independently per batch | `[B,C,C]` | 12 |
| Batched weighted distances | Pairwise point comparison, coordinate weights, coordinate reduction; separate batches | `[B,N,M]` | 15 |
| Multichannel image filters | Valid spatial windows, multiply and reduce over spatial offsets and input channels, multiple output filters | `[B,H-U+1,W-V+1,K]` | 18 |
| Blockwise channel covariance | Partition image blocks, group four observations, center, form every channel pair | `[B,H/2,W/2,C,C]` | 18 |
| Masked diffusion with channel mixing | Three synchronous spatial updates, integer rounding, channel mixing, clipping, frozen cells and borders | `[B,H,W,C]` | 20 |

The total timed allowance is 93 minutes, before practice and breaks. These are
pilot allowances, not estimates of participant completion times. Pilot again
before recruitment; a shorter task subset or two sessions may be needed.

Full participant descriptions in English and Chinese: [tasks.md](tasks.md).
The site highlights order of operations, output shapes, batch separation, filter
orientation, frozen boundaries/cells, and synchronous reads. The searchable
manual includes tested matrix-product, transpose, and batch examples in both
languages.

## Compatibility gate

[Implementation inventory](tasks.py) includes broadcasting, tuple/negative-axis
reductions, keepdims, einsum, batched matmul, per-batch dot and tensordot,
algebraic identities, reshape/transpose, strided and patch slices, masks,
comprehensions, append construction, explicit loops, chained indexing,
default-float allocation with integer-valued results, while counters, copies,
and buffer updates.

Every listed implementation is a required pass. The compiler gate first checks
its ordinary NumPy behavior against an independent scalar oracle, compiles both
the original and an assertion-instrumented program, and executes each compiled
shape on multiple private inputs. Each shape also receives an intentionally
incorrect expected output, which must be rejected. No witness is available at
compilation time.

The separate site gate runs every implementation through the actual sandboxed
participant worker and checks exposed output dimensions and values. Rust
reference solutions run in RISC Zero. There is no Python execution fallback for
participant code. Native NumPy is used only for validation of the inventory and
the server-owned reference calculations.

Fixtures include zero and constant inputs, signed and asymmetric data, randomized
inputs, clipping extremes, zero weights, distinct batches, singleton dimensions,
unequal point counts, rectangular images and filters, whole-height filters,
multiple channels, and grids with no interior cells. Invalid shape and range
inputs are separately rejected at the API boundary.

This is finite compatibility evidence, not a proof that every conceivable valid
Python/NumPy program compiles. New sensible alternatives must be added as required
gate cases; do not remove them or rewrite participant submissions to hide compiler
failures. External libraries and the full NumPy API are not universally supported.

## Repairs discovered during this revision

- Static batched matmul supports broadcasting of leading axes, vector promotion,
  noncontiguous operands, empty batch axes, and zero-length contractions. Function
  `np.matmul` and operator `@` use the same semantics.
- Empty intermediate arrays retain their rank and trailing dimensions through
  conversion, min/max, clipping, selection, and slice assignment.
- Canonical integer while counters with invariant shape bounds are normalized to
  range iteration before symbolic execution. Guards involving changing bounds,
  noninteger limits, break, continue, or return are left on the existing while
  path. Counter increments and post-loop values are preserved.
- For-loop targets preserve their previous values when an enclosing branch is
  inactive, including normalized while counters.
- The study worker supports rank-five outputs and caches compiled programs by
  public input shapes within a run. All differing private inputs still execute
  through the compiled evaluator. Resource limits are unchanged.

Failures before repairs are archived in `before-counted-loop-fix.json` and
`site-before-cache-and-loop-fix.json`; they are intermediate snapshots, not the
complete initial failure inventory.

## Reproduction

From `zinnia-src`:

```sh
RUSTC_BOOTSTRAP=poseidon_primitives,halo2_base PYTHON=.venv/bin/python scripts/develop.sh --offline --features halo2-base-backend,noir-backend
.venv/bin/python -m studies.user_study.revision2.validate --jobs 4 --output studies/user_study/revision2/validation.json
.venv/bin/python -m studies.user_study.revision2.validate_site
.venv/bin/python -m studies.user_study.revision2.validate --variant nested_loops --profile 0 --fixture 6 --backend halo2 --timeout 180 --output studies/user_study/revision2/proofs.json
.venv/bin/python -m studies.user_study.validate --output studies/user_study/revision2/legacy-regression.json
.venv/bin/python -m pytest -q testing/lang/test_revision2_arrays.py testing/lang/test_user_study_array_compat.py
.venv/bin/python -m unittest studies.user_study.site.tests.test_revision2 studies.user_study.site.tests.test_execution studies.user_study.site.tests.test_worker studies.user_study.site.tests.test_manual studies.user_study.site.tests.test_store studies.user_study.site.tests.test_api
npm run test:browser --prefix studies/user_study/site
```

The scoped `RUSTC_BOOTSTRAP` setting matches the earlier validation build on this
workstation: two optional dependencies use nightly-only features. A suitable
nightly toolchain is the alternative.

Site-engine and browser tests need permission to create isolated processes,
network namespaces, and local sockets. They use temporary study databases.
Reports contain compiler, extension, source, and inventory fingerprints. The
real-proof check is a smoke test of one implementation/shape/input per task, not
proof-backend coverage of every inventoried alternative. Participant execution
continues to use compiled Zinnia mock evaluation and RISC Zero execution, without
charging participants for proof generation.

## Study interpretation

The intended manipulation is composition of array operations across several
axes, including managing intermediate shapes and preserving update semantics.
It does not assume that Zinnia will win. Rust may use idiomatic ndarray views,
reductions, matrix products, iterators, or loops. Correctness, completion time,
errors, and assistance remain the main measures; shorter code is supporting
evidence only. Freeze these contracts, compiler build, references, and budgets
after the next pilot and before collecting the revised study data.

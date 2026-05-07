# Array Semantics Testing Plan

## Goal

Build a systematic test suite for Zinnia array semantics, not a larger pile of
toy examples. The test suite should answer three questions for every supported
array operation:

1. Does Zinnia match NumPy for static arrays?
2. Does the dynamic-array implementation match the static implementation where
   they share a semantic surface?
3. When Zinnia intentionally differs from NumPy, is the difference explicit,
   tested, and documented as a compile-time rejection or proof-time assertion?

The canonical implementation target is `zinnia-src`. The older
`zinnia-artifact` tree should be treated as historical context only.

## Testing Principles

### Use NumPy as the semantic oracle

For every operation that is intended to be NumPy-compatible, tests should derive
expected values from NumPy itself whenever practical. Hard-coded expected arrays
are useful for readability in small cases, but they should not be the only
oracle for broad behavior.

Preferred test shape:

```python
cases = [
    np.asarray(...),
    ...
]
for case in cases:
    expected = numpy_expression(case)
    got = zinnia_expression(case)
    assert got == expected
```

Inside Zinnia circuits, the final assertion usually has to be expressed through
`assert`, `.all()`, `.sum()`, or scalar checks. The test driver can still use
NumPy to generate the expected scalar/array values.

### Test static and dynamic parity

For each operation family, maintain three variants when meaningful:

- Static only: `np.asarray(...)`
- Dynamic only: `np.promote_to_dynamic(np.asarray(...))`
- Mixed: one operand static, one operand dynamic

The dynamic variant should not just test that the operation compiles. It should
check the same values, shapes, dtype behavior, axis behavior, and failure modes
as the static variant, subject to the known dynamic-envelope restrictions.

### Separate acceptance from rejection

The suite should explicitly distinguish:

- Supported NumPy semantics that must pass.
- Unsupported NumPy semantics that should fail cleanly at compile time.
- Runtime shape/bound violations that should become proof-time failures.
- Current implementation bugs.

Do not encode "currently panics" as a passing test unless the diagnostic is part
of the intended language behavior.

### Prefer semantic matrices over one-off examples

Each operation family should have a compact matrix over ranks, dtypes, axes, and
edge cases. A good test file should make it obvious which semantic cells are
covered and which are not.

## Current Snapshot

As of this audit, `pytest --collect-only -q testing/operator/ndarray` collects
306 tests. Coverage is broad in names, but much of it is example-level.

Existing test clusters:

- Constructors: `asarray`, `zeros`, `ones`, `arange`, `linspace`
- Static indexing: scalar index, slice, advanced indexing, ellipsis, newaxis
- Dynamic indexing: scalar index, slicing, mask, fancy indexing, setitem
- Element-wise ops: arithmetic, comparisons, selected NumPy ufunc-like calls
- Broadcasting: scalar/array, lower-rank, 2D/3D examples
- Reductions: `sum`, `any`, `all`, `argmax`, `argmin`, `mean`, `var`, `std`,
  `cumsum`, `cumprod`
- Shape manipulation: transpose, stack, concatenate, split, block, flip,
  rotate, squeeze, expand dims, tile, repeat
- Dynamic composition: concat/split/filter/indexing combinations

Main weakness: the tests mostly prove that specific examples work. They do not
yet define a systematic compatibility contract.

## Semantic Dimensions To Cover

### Array representation

Test cells:

- Scalar-like 0D arrays, if supported or intentionally rejected.
- 1D vectors.
- 2D matrices.
- 3D arrays.
- Empty arrays, if supported or intentionally rejected.
- Singleton dimensions: `(1,)`, `(1, n)`, `(n, 1)`, `(1, n, 1)`.
- Ragged input rejection for `np.asarray`.
- Static arrays as circuit inputs.
- Dynamic arrays created by promotion.
- Dynamic arrays as circuit inputs, if the language intends to support them.

Open questions:

- Are 0D arrays meant to be distinct from scalars?
- Are empty arrays valid values, or should all current dynamic envelopes require
  non-empty runtime shapes?
- Should `DynamicNDArray` be allowed at the outer circuit boundary? Older notes
  disagree with examples and parser support, so this needs a settled contract.

### Dtype and scalar promotion

Test cells:

- Integer arrays.
- Float arrays.
- Boolean arrays and boolean-as-integer behavior.
- Mixed integer/float operations.
- Mixed boolean/integer operations.
- Scalar-array promotion.
- Assignment with dtype conversion.
- Concatenate/stack dtype promotion.
- Comparisons returning boolean-like arrays.

Open questions:

- The type layer has a `Boolean` variant, but Python annotation aliases currently
  map `bool`/`Boolean` to `Integer` in places. Tests should pin down whether
  this is intended or transitional.
- Float semantics are circuit-field approximations in several places. Tests
  should document the accepted precision/model instead of relying on incidental
  Python float behavior.

### Indexing and slicing

Test cells:

- Positive and negative scalar indices.
- Out-of-bounds indices: static and dynamic.
- Full slices, omitted bounds, negative bounds.
- Positive and negative steps.
- Dynamic start, stop, and step.
- Multi-axis mixed indexing: scalar + slice, slice + scalar, slice + slice.
- Ellipsis.
- `None` / `np.newaxis`.
- Boolean masks.
- Fancy integer indices.
- Repeated fancy indices.
- Negative fancy indices.
- Multi-dimensional fancy index arrays.
- Paired advanced indexing versus outer/gather semantics.
- Assignment through all supported index forms.

Static/dynamic parity target:

- Static and dynamic indexing should agree on value results.
- Dynamic indexing may use bounded storage and runtime assertions, but should not
  silently change NumPy's indexing meaning.

Known high-risk area:

- Advanced indexing has several subtly different NumPy modes. Boolean masking,
  row gather, paired index arrays, and outer-style index arrays need separate
  contract tests.

### Broadcasting and element-wise ops

Test cells:

- Scalar with array.
- Same-shape arrays.
- Left-padding lower-rank inputs.
- Singleton-axis expansion.
- `(n, 1) op (1, m)` outer-style broadcast.
- 3D broadcast with multiple singleton axes.
- Incompatible shapes.
- Static + dynamic.
- Dynamic + dynamic same shape.
- Dynamic + dynamic singleton broadcast.
- Broadcast followed by reduction.
- Broadcast followed by materializing operation such as setitem or concat.

Static/dynamic parity target:

- For small bounded examples, static and dynamic should produce identical
  values.
- For dynamic cases that would cause unacceptable bound explosion, the compiler
  should refuse with an intentional diagnostic rather than materialize a huge
  envelope.

Known high-risk area:

- The dynamic bound model is currently coarser than the desired design in
  `ROADMAP/04-lazy-views.md`. Tests should expose when an operation silently
  creates a much larger bound than the semantic result needs.

### Reductions and scans

Test cells:

- `axis=None`.
- Every positive axis.
- Negative axes.
- Tuple-of-axes if supported.
- `keepdims` if supported.
- Empty input behavior if arrays can be empty.
- Boolean reductions: `any`, `all`.
- Numeric reductions: `sum`, `prod`, `min`, `max`, `mean`, `var`, `std`.
- Index reductions: `argmax`, `argmin`.
- Scans: `cumsum`, `cumprod` along each axis and flattened.

Static/dynamic parity target:

- Dynamic reductions should agree with static reductions for the same concrete
  runtime values.
- Result rank and shape should be asserted, not only aggregate scalar values.

Known high-risk area:

- Many current tests assert only a final `.sum()`. That can hide wrong ordering,
  wrong shape, or duplicated/missing elements.

### Shape transformations

Test cells:

- `reshape`, including `-1` inference and invalid sizes.
- `transpose` / `.T`, including explicit axes and negative axes.
- `swapaxes`, `moveaxis`.
- `squeeze`, `expand_dims`.
- `flatten`, `flat`, `tolist`.
- `concatenate`, `stack`.
- `vstack`, `hstack`, `dstack`, `column_stack`, `row_stack`.
- `split`, `array_split`, `hsplit`, `vsplit`, `dsplit`.
- `tile`, `repeat`.
- `broadcast_to`.
- `block`.

Static/dynamic parity target:

- Shape metadata should be tested directly where the language exposes shape.
- Value order should be tested with non-commutative expected arrays, not only
  sums.

Known high-risk area:

- Dynamic reshape/transposition may preserve aggregate values while using an
  incorrect flat-index mapping. Tests should use position-sensitive assertions.

### Mutation and aliasing

Test cells:

- Single scalar assignment.
- Slice assignment.
- Row/column assignment.
- Dynamic index assignment.
- Mask assignment.
- Fancy index assignment if supported.
- Scalar RHS broadcasting.
- Array RHS broadcasting.
- Dtype conversion on assignment.
- Assignment followed by multiple reads.
- Assignment to a view, if views become first-class.

Contract questions:

- Does assignment mutate in place, or does the compiler model arrays as fresh
  values with no aliasing?
- If two variables reference the same array value before mutation, should they
  alias like Python, or behave as immutable SSA values?

These questions should be answered by tests before optimizing memory behavior.

### Composition tests

Composition tests should target algebraic interactions rather than random long
programs:

- Slice then broadcast.
- Broadcast then reduce.
- Transpose then index.
- Reshape then setitem.
- Filter then concat.
- Fancy index then broadcast.
- Split then concat round trip.
- Concat then split round trip.
- Dynamic shape operation then reduction.
- Static-to-dynamic promotion in the middle of an expression.

The oracle should check both values and shapes.

## Suggested Test Organization

Create a second-generation suite under one of these paths:

- `testing/operator/ndarray_semantics/`
- or `testing/semantics/array/`

Keep existing tests for regression coverage. New tests should be matrix-driven.

Recommended files:

- `test_static_vs_numpy.py`
- `test_dynamic_vs_static.py`
- `test_indexing_contract.py`
- `test_broadcasting_contract.py`
- `test_shape_contract.py`
- `test_reduction_contract.py`
- `test_mutation_contract.py`
- `test_rejection_contract.py`

Recommended helper module:

- `testing/helpers/array_semantics.py`

Helper responsibilities:

- Build paired static/dynamic circuit variants.
- Generate small deterministic arrays with distinctive values.
- Compare arrays through shape-sensitive assertions.
- Mark expected compile failures separately from expected runtime/proof
  failures.
- Keep NumPy oracle code outside the circuit where possible.

## First Milestone

Milestone 1 should not try to cover all of NumPy. It should establish the
testing pattern and expose the most important semantic gaps.

Scope:

1. Static-vs-NumPy oracle tests for indexing, broadcasting, reshape/transpose,
   and reductions.
2. Dynamic-vs-static parity tests for the same operations on small arrays.
3. Rejection tests for unsupported or intentionally dangerous dynamic cases.
4. Direct shape assertions for every non-scalar array result.
5. A short status table mapping each operation family to:
   `pass`, `bug`, `unsupported`, or `unclear contract`.

Recommended initial cases:

- Broadcasting: `(3,) + (2, 3)`, `(3, 1) + (1, 4)`, incompatible `(2, 3) + (2,)`.
- Indexing: `a[:, i]`, `a[i, :]`, `a[::-1]`, `a[..., 1]`, `a[:, None, :]`.
- Advanced indexing: boolean mask, row gather, paired 2D indices.
- Shape: reshape round trip, transpose with axes, moveaxis.
- Reductions: axis-specific sum/max/argmax with shape checks.
- Mutation: dynamic setitem followed by reads from affected and unaffected cells.

## Success Criteria

This testing track is useful when:

- A new operator can be added by filling rows in an existing semantic matrix.
- Static and dynamic behavior are compared by default.
- Unsupported NumPy behavior is visible as intentional rejection, not accidental
  panic.
- Tests fail on wrong shape/order, not just wrong aggregate.
- The suite gives enough evidence to decide where the dynamic envelope/type
  system is too weak.

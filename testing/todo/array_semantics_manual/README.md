# Array Semantics Manual TODO

These files are the remaining handwritten ndarray semantic tests that have not
been fully retired yet.

They live here as migration backlog, not as active pytest coverage. The
generated suite under `testing/semantics/array/` is the active semantic test
surface.

## Dynamic Composition Migration Notes

First generated slice for `array.dynamic-composition-coverage`:

- `test_dynamic_binary.py::test_chained_ops` is covered by
  `scenario_migration::default::dynamic::int::dynamic_chained_binary_ops`.
- `test_dynamic_indexing.py::test_index_then_binary_op` is covered by
  `scenario_migration::default::dynamic::int::dynamic_index_then_binary_op`.
- `test_dynamic_indexing.py::test_slice_then_sum` is covered by
  `scenario_migration::default::dynamic::int::dynamic_slice_then_sum`.
- `test_dynamic_reshape.py::test_reshape_then_add` is covered by
  `scenario_migration::default::dynamic::int::dynamic_reshape_then_add`.
- `test_dynamic_reshape.py::test_reshape_preserves_sum` is covered by
  `scenario_migration::default::dynamic::int::dynamic_reshape_preserves_sum`.

Measured baseline before this slice: the three parked dynamic composition
files had 66 passing tests and 11 failing tests. The failing residue includes
dynamic-step slicing, fancy indexing, some dynamic slice assignment paths, and
static boolean-mask indexing over dynamic arrays. Those are real compiler gaps,
not generated-suite migration targets for this card.

## Dynamic Join/Split Migration Notes

First generated slice for `array.dynamic-join-split-coverage`:

- `test_dynamic_concat_stack_split.py::test_concat_1d` is covered by
  `scenario_migration::default::dynamic::int::dynamic_concat_1d_sum`.
- `test_dynamic_concat_stack_split.py::test_concat_2d_axis1` is covered by
  `scenario_migration::default::dynamic::int::dynamic_concat_2d_axis1_sum`.
- `test_dynamic_concat_stack_split.py::test_stack_1d_axis0` is covered by
  `scenario_migration::default::dynamic::int::dynamic_stack_axis0_sum`.
- `test_dynamic_concat_stack_split.py::test_split_equal` is covered by
  `scenario_migration::default::dynamic::int::dynamic_split_equal_sections_sum`.
- `test_dynamic_concat_stack_split.py::test_split_then_concat_roundtrip` is
  covered by
  `scenario_migration::default::dynamic::int::dynamic_split_then_concat_roundtrip`.
- `test_dynamic_concat_stack_split.py::test_concat_then_split_roundtrip` is
  covered by
  `scenario_migration::default::dynamic::int::dynamic_concat_then_split_roundtrip`.

Measured baseline before this slice:
`test_dynamic_concat_stack_split.py` is 35/35 passing. Remaining entries in
that parked file are migration residue only; they are still ungenerated
coverage candidates, not known compiler failures from this card.

## Shape/Linspace Migration Notes

First validation/cleanup slice for `array.shape-manipulation-scenario-expansion`:

- `test_linspace.py::test_simple_linspace` is covered by
  `linspace_migration::default::static::int::simple_float_endpoint` and
  `linspace_migration::default::static::int::simple_int_no_endpoint`.
- `test_shape_manipulation.py::test_swapaxes_2d` is covered by
  `shape_manipulation_migration::default::static::int::swapaxes_2d`.
- `test_shape_manipulation.py::test_squeeze_all` is covered by
  `shape_manipulation_migration::default::static::int::squeeze_all`.
- `test_shape_manipulation.py::test_expand_dims_front_and_back` is covered by
  `shape_manipulation_migration::default::static::int::expand_dims_front_and_back`.
- `test_shape_manipulation.py::test_reshape_with_minus_one` is covered by
  `shape_manipulation_migration::default::static::int::reshape_with_minus_one`.

Measured baseline before this slice:
`test_shape_manipulation.py` plus `test_linspace.py` are 36/36 passing. These
parked files are already broadly represented by the generated
`shape_manipulation_migration` and `linspace_migration` families, so this slice
records the first reviewed green subset rather than duplicating manifest cases.

The broad generated filter currently has four failing generated cases:
`linspace_migration::default::static::int::complex_axis_2`,
`linspace_migration::default::static::int::complex_axis_1`,
`linspace_migration::default::static::int::complex_axis_2_no_endpoint`, and
`shape_manipulation_migration::default::static::int::atleast_3d_promotes_1d`.
Those are generated-suite classification residue for a follow-up card, not
compiler changes or blockers for this first validation slice.

## Float Math/Reduction Migration Notes

First generated slice for `array.float-math-generated-coverage`:

- `test_math_and_reductions.py::test_np_sqrt_vectorized` is covered by
  `float_math_migration::default::static::float::float_sqrt_perfect_squares`.
- `test_math_and_reductions.py::test_np_abs_vectorized_2d` is covered by
  `float_math_migration::default::static::float::float_abs_2d`.
- `test_math_and_reductions.py::test_np_negative_vectorized` is covered by
  `float_math_migration::default::static::float::float_negative_1d`.
- `test_math_and_reductions.py::test_floor_basic` is covered by
  `float_math_migration::default::static::float::float_floor_basic`.
- `test_math_and_reductions.py::test_reciprocal_basic` is covered by
  `float_math_migration::default::static::float::float_reciprocal_powers_of_two`.
- `test_math_and_reductions.py::test_mean_no_axis` is covered by
  `float_math_migration::default::static::float::float_mean_no_axis`.

Measured baseline before this slice:
`test_math_and_reductions.py` is 33/33 passing. The first generated slice uses
exact-by-construction float values: perfect squares for `sqrt`, powers of two
for `reciprocal`, and a mean whose result is exactly representable enough for
the current generated assertion renderer.

Remaining handwritten float cases are migration residue or deferred by the
current assertion model. Tolerance-dependent reductions such as `var_axis_0`
and non-exact transcendental/approximate float outputs should wait until the
generated suite can express tolerance-based assertions instead of exact
element/scalar equality.

A file should move out of this TODO area only when one of these is true:

- its behavior is fully migrated into manifest-driven generated cases, or
- the underlying Zinnia behavior is fixed and the generated suite can express
  the regression cleanly as `pass`, `xfail`, or `reject`.

# Revision 4: independent stride and dilation

The current task set keeps the array-relationship tasks introduced in revision 3
and makes a small, targeted increase in task 4: **output stride 2, filter dilation
3** on both spatial axes. With different steps, the previously valid shortcut of
subsampling the entire image by 2 and applying an ordinary filter is no longer
correct. Output shape uses the dilation-3 effective extent and floor division by
the output stride. A hand-worked example explains which input pixels are sampled.

Only the changed contract receives a new ID, `strided_filters_v4`. The other five
IDs retain their existing definitions. All older assignments, including
`strided_filters_v3`, continue to use their original contracts, budgets and grades.

## Fair comparison

Both conditions get identical task statements, public examples, hidden input
cases, time budgets and documentation access. Public feedback remains three
examples per task. Every hint accepts straightforward loops, and Rust may use
broadcasting, views, matrix products, window iterators and helper functions.

The harder relationship is explicit in both English and Chinese. The manual
provides runnable sliding-window examples for both languages and explains
NumPy/ndarray differences in slice bounds and reshaping non-contiguous views.
Difficulty does not rely on withholding those interface rules from Rust users.

Full attempt-trail scoring remains enabled: each accepted example/custom run,
including compilation errors and repeated code, has immutable source and timing
metadata and a linked private evaluation. Ratings and pauses do not add to active
time. Evaluation happens after completion/withdrawal; hidden scores do not become
participant feedback. The researcher can compare each attempt with final
submission. See [the attempt-trail protocol](../revision3/README.md).

This design makes shape, indexing, layout and update mistakes observable. It does
not assume that Rust must lose. Participant error rates and repair time still
need to be measured.

## Required implementation coverage

The current inventory has **67 Zinnia implementations**:

- The previous 49 implementations for the five unchanged tasks.
- 18 implementations of the changed filter task: scalar/chained/while loops,
  patch products, offset accumulation, broadcasting, einsum, dot/matmul,
  tensordot, sliding-window views, comprehensions, explicit helper functions,
  integer and default-float allocation, an explicitly filled empty buffer,
  and common sliding-window imports.

The compiler gate covers 219 implementation/shape combinations, eight private
input fixtures each: **1,752 correct-output checks and 219 incorrect-output
controls**. It compiles the original source and an assertion-instrumented source,
then reuses each compiled program for changing private inputs.

The participant-engine gate runs all 67 Zinnia implementations plus **22 valid
Rust alternatives**, including fully broadcast implementations of projected
distances and block covariance. They run on all fixtures across all profiles:
**2,312 actual engine checks**. Zinnia uses the compiled IR evaluator; Rust uses
RISC Zero. No participant Python execution fallback is used.

Four filter profiles include rectangular windows, unequal channels/filter counts,
multiple batches, singleton dimensions, leftover pixels, and a filter whose
receptive field exactly fits the image. The hidden-case count for task 4 is 11,
identical for both conditions. Other tasks retain their existing nine hidden cases.

Separate matched negative controls show that both engines reject the old
subsampling shortcut and an implementation using dilation 2 instead of 3. These
programs compile and execute; they fail because they compute the wrong function.

No failing valid alternative is excluded. This finite inventory is evidence,
not proof that every imaginable valid NumPy program compiles. Newly reported
sensible alternatives remain mandatory regression cases.

## Compatibility repairs

The existing compiler supports sliding-window views, but the import normalizer
previously rejected `from numpy.lib.stride_tricks import sliding_window_view`
and module aliases. It now resolves the supported submodules/members to the same
canonical NumPy operation. Lexical shadowing and restrictions on other imports
are preserved and regression-tested.

The revision-4 correctness wrapper now instruments only kernel returns, leaving
nested helper returns intact. This fixes a validator limitation that could
incorrectly reject a correct implementation returning intermediate arrays from a
helper. The participant worker already handled this correctly.

## Reproduction and reports

From `zinnia-src`:

```sh
.venv/bin/python -m studies.user_study.revision4.validate --jobs 4 --output studies/user_study/revision4/validation.json
.venv/bin/python -m studies.user_study.revision4.validate_site
.venv/bin/python -m studies.user_study.revision4.validate --jobs 2 --variant nested_loops --profile 0 --fixture 6 --backend halo2 --timeout 180 --output studies/user_study/revision4/proofs.json
.venv/bin/python -m pytest -q testing/lang/test_function_lowering.py testing/lang/test_revision2_arrays.py testing/lang/test_user_study_array_compat.py
.venv/bin/python -m unittest studies.user_study.site.tests.test_revision4 studies.user_study.site.tests.test_attempt_trail studies.user_study.site.tests.test_api studies.user_study.site.tests.test_manual
npm run test:browser --prefix studies/user_study/site
```

Reports: [compiler gate](validation.json), [participant engines](site-validation.json),
[proof smoke checks](proofs.json). Real Halo2 proof smoke checks cover one
implementation/shape/random fixture per task; they do not claim proof-backend
coverage for every implementation. Engine/API/browser tests use temporary data.
Each of the six formal tasks has a 10-minute budget (60 minutes total). Tutorials, untimed practice, pauses and rating time are additional.

Full participant text: [tasks.md](tasks.md).

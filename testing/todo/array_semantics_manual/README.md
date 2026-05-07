# Array Semantics Manual TODO

These files are the remaining handwritten ndarray semantic tests that have not
been fully retired yet.

They live here as migration backlog, not as active pytest coverage. The
generated suite under `testing/semantics/array/` is the active semantic test
surface.

A file should move out of this TODO area only when one of these is true:

- its behavior is fully migrated into manifest-driven generated cases, or
- the underlying Zinnia behavior is fixed and the generated suite can express
  the regression cleanly as `pass`, `xfail`, or `reject`.

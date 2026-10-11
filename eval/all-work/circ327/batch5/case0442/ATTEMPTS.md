# case0442

Category: `accepted_with_rewrite`

Stable masked compaction into capacity 20, with logical length.

as_written: not expressible

C = A[~np.in1d(A,B)]

rewrite: accepted

Compilation accepted.

- Replace boolean-indexed output with capacity 20, masked stable compaction, and public logical length; at most all 20 entries of A survive.

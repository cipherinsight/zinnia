# case0509

Category: `accepted_with_rewrite`

Bounded min/max and padded 10x10 crop with logical dimensions.

as_written: not expressible

rows/cols = np.flatnonzero(...), and value-dependent crop at source lines 12-18

rewrite: accepted

Compilation accepted.

- Replace variable-length nonzero-index lists by guarded min/max reductions over the 10 rows/columns.
- Pad the crop to a 10x10 rectangle with public logical height and width; each dimension of a crop of a 10x10 input is at most 10.

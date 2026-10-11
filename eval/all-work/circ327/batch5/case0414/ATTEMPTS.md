# case0414

Category: `accepted_with_rewrite`

Eight-bin capacity; guarded sums and native integer division by bin size.

as_written: not expressible

data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size), line 11

rewrite: accepted

Compilation accepted.

- Pad the result to eight bins with a public logical count; at most eight nonempty bins fit in eight input elements.
- Replace dynamic reshape and row means by guarded static reductions over the eight input elements. Division of a Q32 sum by a positive integer uses CirC native unsigned integer division with sign conversion; it requires no wide Q32-by-Q32 division gadget.

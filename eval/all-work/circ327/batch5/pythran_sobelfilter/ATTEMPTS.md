# pythran_sobelfilter

Category: `accepted_with_rewrite`

Bounds follow from valid 16-element accesses; collapse repeated zero-column writes.

as_written: not expressible

for i in range(rows); for i in range(1, cols-1); nested ranges(rows-1, cols-1), lines 11-19

rewrite: accepted

Compilation accepted.

- For valid executions with cols != 0, the first loop has at most 16 iterations because its edge-image accesses must lie in [-16,15]; replace by a 16-step guarded loop.
- For cols == 0, collapse repetitions of the same two boundary writes into one guarded pair; this preserves arbitrary positive rows without inventing a bound.
- The second loop has at most 16 iterations because edge_image[i] must be in bounds; inner image loops are bounded by the first two loops. Use guarded iterations and normalize Python negative indices.

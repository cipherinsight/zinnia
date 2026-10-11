Program: case0419

As written: not_expressible
kernel.py:12 reshape/bin count is private-dependent; capacity 8x6 from input shape.

Minimal rewrite: accepted
Compiled and checked on both fixtures.

- Pad the output to 8x6 and publish its logical column count; each bin consumes at least one of 6 input columns, while bin_size>6 gives zero bins.
- Replace the private reshape with guarded address arithmetic; divide the signed Q32 sum by its integer count using stock u64 division (no rescaling and no handwritten division circuit).

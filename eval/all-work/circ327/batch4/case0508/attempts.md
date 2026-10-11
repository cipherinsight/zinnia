Program: case0508

As written: not_expressible
kernel.py:11 argwhere and line 13 crop have data-dependent shapes.

Minimal rewrite: accepted
Compiled and checked on both fixtures.

- Fuse argwhere coordinate materialization with guarded min/max reductions; pad cropped output to 10x10 plus logical dimensions, bounded by the input shape. All-zero input is invalid in NumPy (empty min).
- Pack buffer and length/dimensions into one public return array.

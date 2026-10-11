Program: case0413

As written: not_expressible
kernel.py:11 boolean filter has value-dependent length.

Minimal rewrite: accepted
Compiled and checked on both fixtures.

- Encode each complex number as a pair of Q32 integers; pad filtered output to 10 pairs and return length, bounded by the input length.
- Pack buffer and length/dimensions into one public return array.

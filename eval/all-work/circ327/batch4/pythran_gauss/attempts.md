Program: pythran_gauss

As written: not_expressible
kernel.py:29 private n loop bound.

Minimal rewrite: needs_gadget
kernel.py:32 reciprocal of a private pivot requires fixed-point division.

- Guard loops over 4 rows; valid indexed n cannot exceed 4, and the explicit domain also gives n<=1024. Reciprocal remains required.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

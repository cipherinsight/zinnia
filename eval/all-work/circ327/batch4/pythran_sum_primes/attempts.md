Program: pythran_sum_primes

As written: not_expressible
kernel.py:27 value-dependent range and filtered list; helper line 17 while.

Minimal rewrite: needs_gadget
kernel.py:15 math.sqrt(n) and ceil require a square-root gadget.

- n<=1024 bounds candidate and divisor loops by 1024; mask the sum instead of building the filtered list. Square root remains required.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

Program: pythran_primes_sieve2

As written: not_expressible
kernel.py:18 dynamic list and lines 23,27 while loops.

Minimal rewrite: needs_gadget
kernel.py:19 n**0.5 requires square root; do not replace the source algorithm by a different sieve.

- n<=1024 gives capacity 511 odd candidates, at most 511 outer steps and 511 inner steps; pad and guard. Square root remains required.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

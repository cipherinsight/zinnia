Program: pythran_vibr_energy

As written: needs_gadget
kernel.py:10 numpy.exp requires fixed-point exponential.

Minimal rewrite: needs_gadget
kernel.py:10 numpy.exp requires fixed-point exponential.

- No allowed local rewrite removes this obstruction without changing the domain or implementing a prohibited gadget.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

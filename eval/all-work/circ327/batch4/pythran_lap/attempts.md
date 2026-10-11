Program: pythran_lap

As written: needs_gadget
kernel.py:25 lap/dh2 requires fixed-point division by private dh2.

Minimal rewrite: needs_gadget
kernel.py:25 lap/dh2 requires fixed-point division by private dh2.

- No allowed local rewrite removes this obstruction without changing the domain or implementing a prohibited gadget.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

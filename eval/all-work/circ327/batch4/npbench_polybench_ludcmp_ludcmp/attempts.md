Program: npbench_polybench_ludcmp_ludcmp

As written: needs_gadget
kernel.py:15,21 division by private matrix entries requires fixed-point division.

Minimal rewrite: needs_gadget
kernel.py:15,21 division by private matrix entries requires fixed-point division.

- No allowed local rewrite removes this obstruction without changing the domain or implementing a prohibited gadget.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

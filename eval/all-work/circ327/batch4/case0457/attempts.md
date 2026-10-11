Program: case0457

As written: needs_gadget
kernel.py:11 np.linalg.norm requires fixed-point square root.

Minimal rewrite: needs_gadget
kernel.py:11 np.linalg.norm requires fixed-point square root.

- No allowed local rewrite removes this obstruction without changing the domain or implementing a prohibited gadget.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

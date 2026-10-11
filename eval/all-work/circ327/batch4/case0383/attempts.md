Program: case0383

As written: not_expressible
kernel.py:11 slice [:N] has input-dependent length.

Minimal rewrite: needs_gadget
kernel.py:11 np.argsort requires sort indices.

- Pad sliced output to 100 entries with Python slice length; input a has 100 elements. Sorting remains required.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

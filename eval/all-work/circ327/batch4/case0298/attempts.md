Program: case0298

As written: not_expressible
kernel.py:11-12 unique-dependent vals.size and output width.

Minimal rewrite: needs_gadget
kernel.py:11 np.unique sorts values and returns inverse ranks; requires a sorting/unique gadget.

- At most 5 unique values: pad vals to 5 and b to 5x5 with logical width; sorting remains required.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

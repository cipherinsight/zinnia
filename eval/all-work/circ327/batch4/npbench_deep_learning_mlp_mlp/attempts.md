Program: npbench_deep_learning_mlp_mlp

As written: needs_gadget
kernel.py:15 softmax np.exp requires fixed-point exp; line 17 also requires division.

Minimal rewrite: needs_gadget
kernel.py:15 softmax np.exp requires fixed-point exp; line 17 also requires division.

- No allowed local rewrite removes this obstruction without changing the domain or implementing a prohibited gadget.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

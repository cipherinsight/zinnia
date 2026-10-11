Program: pythran_convnet

As written: not_expressible
kernel.py:17,20 private-dependent reshape and allocations.

Minimal rewrite: needs_gadget
kernel.py:18 tanh and helper line 10 exp/division require gadgets.

- Nonempty valid reshape fixes length*batch_size*state_size=4, bounding positive dimensions by 4; pad and guard. tanh/exp remain required.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

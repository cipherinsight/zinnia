Program: pythran_allpairs_distances_loops

As written: not_expressible
kernel.py:19-20 ones((500,d)), ones((200,d)) use unbounded d; do not algebraically replace the kernel by zeros.

Minimal rewrite: not_expressible
kernel.py:19-20 ones((500,d)), ones((200,d)) use unbounded d; do not algebraically replace the kernel by zeros.

- No allowed local rewrite removes this obstruction without changing the domain or implementing a prohibited gadget.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

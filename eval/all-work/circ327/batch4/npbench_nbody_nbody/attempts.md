Program: npbench_nbody_nbody

As written: not_expressible
kernel.py:61-62 energy arrays have length Nt+1 and line 65 loop uses unbounded Nt; bound on N does not bound Nt.

Minimal rewrite: not_expressible
kernel.py:61-62 energy arrays have length Nt+1 and line 65 loop uses unbounded Nt; bound on N does not bound Nt.

- No allowed local rewrite removes this obstruction without changing the domain or implementing a prohibited gadget.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

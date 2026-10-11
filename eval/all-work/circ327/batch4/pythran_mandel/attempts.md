Program: pythran_mandel

As written: not_expressible
kernel.py:11 output shape depends on size; line 18 while loop depends on max_iteration.

Minimal rewrite: not_expressible
kernel.py:18 while loop has private max_iteration without upper bound; size<=1024 bounds only image shape.

- Pad image to 1024x1024 using the size contract; no finite iteration bound follows for max_iteration.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

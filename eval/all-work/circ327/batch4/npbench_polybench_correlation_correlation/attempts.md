Program: npbench_polybench_correlation_correlation

As written: not_expressible
kernel.py:15 np.eye(M) and line 16 loop depend on private M.

Minimal rewrite: needs_gadget
kernel.py:11,14 standard deviation and sqrt(float_n) require square root and division.

- Valid indexed columns imply M<=14 except degenerate nonpositive sizes; capacity 14 and guarded loops suffice; sqrt remains required.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

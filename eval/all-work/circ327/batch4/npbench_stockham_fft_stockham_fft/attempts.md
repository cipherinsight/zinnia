Program: npbench_stockham_fft_stockham_fft

As written: not_expressible
kernel.py:11-12 arrays depend on private R; line 16 loop depends on K.

Minimal rewrite: not_expressible
kernel.py:11-12 allocate R by R with unbounded R when K=0 (valid, skips all FFT iterations); complex exp also needs a gadget.

- For K>=1, reshape validity R**K=16 bounds R<=16 and K<=4; K<=0 skips the loop but unbounded R still allocates the unused DFT. No complete bounded rewrite; complex exponential also missing.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.

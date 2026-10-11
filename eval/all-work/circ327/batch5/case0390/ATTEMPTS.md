# case0390

Category: `timeout_or_oom`

20,000-element padded patch scatter exceeded the 8 GB cap.

as_written: not expressible

x = a[:a.shape[0] // patch_size * patch_size, ...]; dynamic reshape on lines 11-12

rewrite: out of memory

Running frontend
Running IR optimizations

- Use a flat padded output capacity 20000, with public patch count and patch side; retained patches contain no more elements than the 100x200 input.
- Guard source elements outside whole patches and scatter to the exact transpose/reshape order. Treat patch_size > 100 as empty output without squaring the unbounded side.

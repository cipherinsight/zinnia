# DS-1000 case 0390 — handwritten Noir skipped

The kernel uses a runtime patch_size to drive a nested-reshape index
permutation:

```python
x = a[:H//p*p, :W//p*p]
result = x.reshape(H//p, p, W//p, p).swapaxes(1, 2).transpose(...).reshape(-1, p, p)
```

Noir's for-loop bounds must be compile-time, and the per-cell index
mapping is a function of `patch_size`. The original handwritten baseline
fell back on `assert(patch_size == PS)` to specialize fully on the
sample value (PS = 4 / 10) — a cheat: it proves a weaker statement than
the Python kernel admits, since any valid divisor of (H, W) should work.

The honest alternative — iterate over the max patch grid and runtime-
mask — would expand to ~200M loop iterations for shape (100, 200),
which Noir can compile but which would obliterate the comparison's
point. The case is genuinely **not naturally expressible** in
handwritten Noir without specializing on the sample value, so we drop
it from the Zinnia-vs-handwritten sweep rather than keep a
specialization-cheat baseline. `main.nr.skipped_specialization`
preserves the original for the record.

Zinnia handles the kernel as-is (its IR emits the per-cell mux mask for
the masked slice + the static permutation for the reshape chain).

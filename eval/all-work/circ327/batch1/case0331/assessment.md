# case0331

Source: kernel.py:11

np.gcd expands to a Euclidean remainder loop; unconstrained Integer operands give no declared iteration bound.

As-written: not expressible.
Minimal rewrite: not_expressible: dynamic loop.

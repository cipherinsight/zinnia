# case0498

Source: kernel.py:11

np.pad output dimensions come from unconstrained Integer[2] shape; source dimensions do not bound padding above.

As-written: not expressible.
Minimal rewrite: not_expressible: dynamic shape.

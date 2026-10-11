# case0431

Source: kernel.py:11

np.std computes a square root of a data-dependent variance; fixed-point sqrt is not supplied.

As-written: needs gadget.
Minimal rewrite: needs_gadget: sqrt.

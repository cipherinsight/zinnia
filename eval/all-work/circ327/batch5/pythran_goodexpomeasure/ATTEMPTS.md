# pythran_goodexpomeasure

Category: `needs_gadget: exp`

Needs fixed-point exp and division by private sigma; cannot precompute from fixtures.

as_written: needs gadget

kernel.py:17: goodExpoR = numpy.exp(-(R - 128) ** 2 / sigma). Needs fixed-point exp and division by private sigma; cannot precompute from fixtures.

rewrite: needs gadget

Needs fixed-point exp and division by private sigma; cannot precompute from fixtures.


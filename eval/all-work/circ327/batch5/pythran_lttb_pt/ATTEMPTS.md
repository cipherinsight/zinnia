# pythran_lttb_pt

Category: `needs_gadget: division`

Valid n_out lies in [3,16], giving capacity 16 and at most 14 bins. Native integer division can compute bin means, but triangle areas multiply Q32 differences; full-width products and their Q32 rescaling need a custom wide fixed-point arithmetic gadget after native arbitrary-integer lowering fails. Guarded padding alone cannot supply that arithmetic.

as_written: not expressible

kernel.py:36: c = next_bin.mean(axis=0). Valid n_out lies in [3,16], giving capacity 16 and at most 14 bins. Native integer division can compute bin means, but triangle areas multiply Q32 differences; full-width products and their Q32 rescaling need a custom wide fixed-point arithmetic gadget after native arbitrary-integer lowering fails. Guarded padding alone cannot supply that arithmetic.

rewrite: needs gadget

Valid n_out lies in [3,16], giving capacity 16 and at most 14 bins. Native integer division can compute bin means, but triangle areas multiply Q32 differences; full-width products and their Q32 rescaling need a custom wide fixed-point arithmetic gadget after native arbitrary-integer lowering fails. Guarded padding alone cannot supply that arithmetic.


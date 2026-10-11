# case0366

Source: kernel.py:12

np.insert with an arbitrary position vector uses stable sorting of the adjusted insertion positions; a custom sorting gadget is prohibited. Output capacity itself is static (105 rows).

As-written: needs gadget.
Minimal rewrite: needs_gadget: sort.

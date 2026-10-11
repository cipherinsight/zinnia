# case0384

Category: `needs_gadget: matrix inverse`

Signed private n includes negative powers, requiring matrix inversion and real division; no native fixed-point matrix-power/inverse routine exists. Repeated powers also have no declared exponent bound.

as_written: needs gadget

kernel.py:11: result = np.linalg.matrix_power(A, n). Signed private n includes negative powers, requiring matrix inversion and real division; no native fixed-point matrix-power/inverse routine exists. Repeated powers also have no declared exponent bound.

rewrite: needs gadget

Signed private n includes negative powers, requiring matrix inversion and real division; no native fixed-point matrix-power/inverse routine exists. Repeated powers also have no declared exponent bound.


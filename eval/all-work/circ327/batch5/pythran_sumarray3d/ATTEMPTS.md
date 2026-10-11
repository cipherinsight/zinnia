# pythran_sumarray3d

Category: `needs_gadget: sqrt`

Both returned arrays must be retained; the first requires inverse square roots. Returning only sum_array3d would weaken the program.

as_written: needs gadget

kernel.py:27: dr = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz). Both returned arrays must be retained; the first requires inverse square roots. Returning only sum_array3d would weaken the program.

rewrite: needs gadget

Both returned arrays must be retained; the first requires inverse square roots. Returning only sum_array3d would weaken the program.


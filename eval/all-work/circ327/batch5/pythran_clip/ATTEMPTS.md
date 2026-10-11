# pythran_clip

Category: `needs_gadget: sqrt`

Complex absolute value needs sqrt(real^2+imag^2), and limit1 divides by that magnitude; complex numbers could otherwise be represented by real/imaginary pairs.

as_written: needs gadget

kernel.py:19: if abs(z[i]) > _max:. Complex absolute value needs sqrt(real^2+imag^2), and limit1 divides by that magnitude; complex numbers could otherwise be represented by real/imaginary pairs.

rewrite: needs gadget

Complex absolute value needs sqrt(real^2+imag^2), and limit1 divides by that magnitude; complex numbers could otherwise be represented by real/imaginary pairs.


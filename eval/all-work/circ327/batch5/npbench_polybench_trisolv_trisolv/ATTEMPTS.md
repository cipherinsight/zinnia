# npbench_polybench_trisolv_trisolv

Category: `needs_gadget: division`

Triangular solve divides each computed real numerator by private L[i,i]; native field division is modular inverse, not Q32 real division.

as_written: needs gadget

kernel.py:11: x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]. Triangular solve divides each computed real numerator by private L[i,i]; native field division is modular inverse, not Q32 real division.

rewrite: needs gadget

Triangular solve divides each computed real numerator by private L[i,i]; native field division is modular inverse, not Q32 real division.


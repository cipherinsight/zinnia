# pythran_crotg

Source: kernel.py:15

Complex magnitude and NORM require fixed-point sqrt, followed by division.

As-written: needs gadget.
Minimal rewrite: needs_gadget: sqrt.

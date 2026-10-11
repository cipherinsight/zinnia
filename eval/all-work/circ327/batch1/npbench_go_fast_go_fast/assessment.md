# npbench_go_fast_go_fast

Source: kernel.py:12

np.tanh on private diagonal elements requires a fixed-point transcendental routine.

As-written: needs gadget.
Minimal rewrite: needs_gadget: tanh.

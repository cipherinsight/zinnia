# case0325

Source: kernel.py:11

np.sin(np.deg2rad(number)) and np.sin(number) need fixed-point sine; no supplied Z# implementation.

As-written: needs gadget.
Minimal rewrite: needs_gadget: sin.

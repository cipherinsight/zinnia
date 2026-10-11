# case0438

Source: kernel.py:11

np.corrcoef normalizes covariance by data-dependent standard deviations, requiring sqrt and fixed-point division.

As-written: needs gadget.
Minimal rewrite: needs_gadget: sqrt.

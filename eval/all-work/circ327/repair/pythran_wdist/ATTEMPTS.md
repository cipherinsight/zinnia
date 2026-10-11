kernel.py:16 uses np.sqrt((wdiff ** 2).sum()); native Q32 division repairs the division subexpression but cannot supply square root. All loops already have static bounds.
No dummy arithmetic-only program is counted as the kernel.

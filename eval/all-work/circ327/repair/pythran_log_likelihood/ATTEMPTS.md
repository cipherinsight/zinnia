kernel.py:12,14 use numpy.exp(-s) and numpy.log(pdfs). Native unsigned division handles the arithmetic, but these transcendental calls have no native route.
No dummy arithmetic-only program is counted as the kernel.

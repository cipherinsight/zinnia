from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: complex, denominator-obligation
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Diagnostic: Complex kernels require Complex dtype support or a fail-closed classification.
import math, numpy as np

@zk_circuit
def fft(x: Complex[Array, 'N<=1024']):
    N = x.shape[0]
    if N == 1:
        return np.array(x)
    e = fft(x[::2])
    o = fft(x[1::2])
    M = N // 2
    l = [e[k] + o[k] * math.e ** (-2j * math.pi * k / N) for k in range(M)]
    r = [e[k] - o[k] * math.e ** (-2j * math.pi * k / N) for k in range(M)]
    return np.array(l + r)

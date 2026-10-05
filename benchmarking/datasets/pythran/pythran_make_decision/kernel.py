from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: complex, in-place-mutation, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: Complex kernels require Complex dtype support or a fail-closed classification.
import numpy as np

@zk_circuit
def md(E: Complex[Array, 'N<=1024'], symbols: Complex[Array, 'N<=1024']):
    L = E.shape[0]
    M = symbols.shape[0]
    syms_out = np.zeros(L, dtype=E.dtype)
    for i in range(L):
        im = np.argmin(abs(E[i] - symbols) ** 2)
        syms_out[i] = symbols[im]
    return syms_out

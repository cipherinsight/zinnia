from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: complex, domain-obligation, in-place-mutation, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: Complex kernels require Complex dtype support or a fail-closed classification.
import numpy as np

@zk_circuit
@requires(lambda xmin, xmax, ymin, ymax, xn, yn, itermax, horizon: 0 <= itermax <= 1024)
def mandelbrot(xmin: Float[Array, 'N<=1024'], xmax: Float[Array, 'N<=1024'], ymin: Float[Array, 'N<=1024'], ymax: Float[Array, 'N<=1024'], xn: Complex[Array, 'N<=1024'], yn: Complex[Array, 'N<=1024'], itermax: Integer, horizon: Float=2.0):
    Xi, Yi = np.mgrid[0:xn, 0:yn]
    X = np.linspace(xmin, xmax, xn, dtype=np.float64)[Xi]
    Y = np.linspace(ymin, ymax, yn, dtype=np.float64)[Yi]
    C = X + Y * 1j
    N_ = np.zeros(C.shape, dtype=np.int64)
    Z_ = np.zeros(C.shape, dtype=np.complex128)
    Xi.shape = Yi.shape = C.shape = xn * yn
    Z = np.zeros(C.shape, np.complex128)
    for i in range(itermax):
        if not len(Z):
            break
        np.multiply(Z, Z, Z)
        np.add(Z, C, Z)
        I = abs(Z) > horizon
        N_[Xi[I], Yi[I]] = i + 1
        Z_[Xi[I], Yi[I]] = Z[I]
        np.logical_not(I, I)
        Z = Z[I]
        Xi, Yi = (Xi[I], Yi[I])
        C = C[I]
    return (Z_.T, N_.T)

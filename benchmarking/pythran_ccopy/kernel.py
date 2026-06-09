from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: complex, in-place-mutation, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: Complex kernels require Complex dtype support or a fail-closed classification.

@zk_circuit
@requires(lambda N, CX, INCX, CY, INCY: 0 <= N <= 1024)
def CCOPY(N: Integer, CX: Complex[Array, 'N<=1024'], INCX: Integer, CY: Complex[Array, 'N<=1024'], INCY: Integer):
    if N <= 0:
        return
    if INCX == 1 and INCY == 1:
        for I in range(N):
            CY[I] = CX[I]
    else:
        IX = 0
        IY = 0
        if INCX < 0:
            IX = (-N + 1) * INCX
        if INCY < 0:
            IY = (-N + 1) * INCY
        for I in range(N):
            CY[IY] = CX[IX]
            IX = IX + INCX
            IY = IY + INCY
    return

from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: complex, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Diagnostic: Complex kernels require Complex dtype support or a fail-closed classification.

@zk_circuit
@requires(lambda N, CX, INCX, CY, INCY: 0 <= N <= 1024)
def CDOTC(N: Integer, CX: Complex[Array, 'N<=1024'], INCX: Integer, CY: Complex[Array, 'N<=1024'], INCY: Integer):
    CTEMP = complex(0.0, 0.0)
    CDOTC = complex(0.0, 0.0)
    if N <= 0:
        return
    if INCX == 1 and INCY == 1:
        for I in range(N):
            CTEMP = CTEMP + CX[I].conjugate() * CY[I]
    else:
        IX = 0
        IY = 0
        if INCX < 0:
            IX = (-N + 1) * INCX
        if INCY < 0:
            IY = (-N + 1) * INCY
        for I in range(N):
            CTEMP = CTEMP + CX[IX].conjugate() * CY[IY]
            IX = IX + INCX
            IY = IY + INCY
    return CTEMP

# Source: Pythran tests/cases/cdotc.py
# Original #pythran export: CDOTC(int, complex list, int, complex list, int)
# Migration notes: complex types likely unsupported.
from zinnia import *


@zk_circuit
def CDOTC(N: int, CX: NDArray[Complex, 64], INCX: int, CY: NDArray[Complex, 64], INCY: int):
    CTEMP = complex(0.0, 0.0)
    CDOTC = complex(0.0, 0.0)
    if (N <= 0):
        pass
    if (INCX == 1 and INCY == 1):
        for I in range(N):
            CTEMP = CTEMP + (CX[I].conjugate()) * CY[I]
    else:
        IX = 0
        IY = 0
        if (INCX < 0):
            IX = (-N + 1) * INCX
        if (INCY < 0):
            IY = (-N + 1) * INCY
        for I in range(N):
            CTEMP = CTEMP + (CX[IX].conjugate()) * CY[IY]
            IX = IX + INCX
            IY = IY + INCY
    _zinnia_result = CTEMP

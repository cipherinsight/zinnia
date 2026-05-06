# Source: Pythran tests/cases/caxpy.py
# Original #pythran export: CAXPY(int, complex, complex list, int, complex list, int)
# Migration notes: complex types likely unsupported.
from zinnia import *


@zk_circuit
def CAXPY(N: int, CA: complex, CX: list, INCX: int, CY: list, INCY: int):
    if N <= 0:
        pass
    if (abs(CA) == 0.0E+0):
        pass
    if (INCX == 1 and INCY == 1):
        for I in range(N):
            CY[I] = CY[I] + CA * CX[I]
    else:
        IX = 0
        IY = 0
        if (INCX < 0):
            IX = (-N + 1) * INCX
        if (INCY < 0):
            IY = (-N + 1) * INCY
        for I in range(N):
            CY[IY] = CY[IY] + CA * CX[IX]
            IX = IX + INCX
            IY = IY + INCY
    _zinnia_result = CY

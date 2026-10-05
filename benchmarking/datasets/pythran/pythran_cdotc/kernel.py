# pythran program cdotc.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
@requires(lambda N, CX, INCX, CY, INCY: 0 <= N <= 1024)
def CDOTC(N: Integer, CX: Complex[Array, '4'], INCX: Integer, CY: Complex[Array, '4'], INCY: Integer):
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

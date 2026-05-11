# Source: NPBench polybench/k3mm (k3mm_numpy.py)
# Original signature: kernel(A, B, C, D)
#   where A (NI,NK), B (NK,NJ), C (NJ,NM), D (NM,NL).
# Migration notes:
#   - NI, NJ, NK, NL, NM hoisted as module-level shape constants.
from zinnia import *

NI = 800
NJ = 850
NK = 900
NL = 950
NM = 1000


@zk_circuit
def k3mm(A: NDArray[Float, 800, 900],
         B: NDArray[Float, 900, 850],
         C: NDArray[Float, 850, 1000],
         D: NDArray[Float, 1000, 950]):
    _zinnia_result = A @ B @ C @ D

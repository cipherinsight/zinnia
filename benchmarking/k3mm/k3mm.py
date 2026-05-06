# Source: NPBench polybench/k3mm (k3mm_numpy.py)
# Original signature: kernel(A, B, C, D)
#   where A (NI,NK), B (NK,NJ), C (NJ,NM), D (NM,NL).
# Migration notes:
#   - NI, NJ, NK, NL, NM hoisted as module-level shape constants.
from zinnia import *

NI = 8
NJ = 8
NK = 8
NL = 8
NM = 8


@zk_circuit
def k3mm(A: NDArray[Float, 8, 8],
         B: NDArray[Float, 8, 8],
         C: NDArray[Float, 8, 8],
         D: NDArray[Float, 8, 8]):
    _zinnia_result = A @ B @ C @ D

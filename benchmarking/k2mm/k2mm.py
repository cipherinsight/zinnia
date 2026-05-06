# Source: NPBench polybench/k2mm (k2mm_numpy.py)
# Original signature: kernel(alpha, beta, A, B, C, D)
#   where A is (NI,NK), B (NK,NJ), C (NJ,NL), D (NI,NL).
# Migration notes:
#   - NI, NJ, NK, NL hoisted as module-level shape constants.
#   - alpha, beta kept as float params.
from zinnia import *

NI = 8
NJ = 8
NK = 8
NL = 8


@zk_circuit
def k2mm(alpha: float, beta: float,
         A: NDArray[Float, 8, 8],
         B: NDArray[Float, 8, 8],
         C: NDArray[Float, 8, 8],
         D: NDArray[Float, 8, 8]):
    D[:] = alpha * A @ B @ C + beta * D

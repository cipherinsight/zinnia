# Source: NPBench polybench/k2mm (k2mm_numpy.py)
# Original signature: kernel(alpha, beta, A, B, C, D)
#   where A is (NI,NK), B (NK,NJ), C (NJ,NL), D (NI,NL).
# Migration notes:
#   - NI, NJ, NK, NL hoisted as module-level shape constants.
#   - alpha, beta kept as float params.
from zinnia import *

NI = 800
NJ = 850
NK = 900
NL = 950


@zk_circuit
def k2mm(alpha: float, beta: float,
         A: NDArray[Float, 800, 900],
         B: NDArray[Float, 900, 850],
         C: NDArray[Float, 850, 950],
         D: NDArray[Float, 800, 950]):
    D[:] = alpha * A @ B @ C + beta * D

# Source: NPBench polybench/doitgen (doitgen_numpy.py)
# Original signature: kernel(NR, NQ, NP, A, C4) where A is (NR, NQ, NP), C4 is (NP, NP).
# Migration notes:
#   - NR, NQ, NP hoisted as module-level shape constants.
from zinnia import *

NR = 8
NQ = 8
NP = 8


@zk_circuit
def doitgen(A: NDArray[Float, 8, 8, 8], C4: NDArray[Float, 8, 8]):
    A[:] = np.reshape(np.reshape(A, (8, 8, 1, 8)) @ C4, (8, 8, 8))

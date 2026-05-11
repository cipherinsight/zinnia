# Source: NPBench polybench/doitgen (doitgen_numpy.py)
# Original signature: kernel(NR, NQ, NP, A, C4) where A is (NR, NQ, NP), C4 is (NP, NP).
# Migration notes:
#   - NR, NQ, NP hoisted as module-level shape constants.
from zinnia import *

NR = 60
NQ = 60
NP = 128


@zk_circuit
def doitgen(A: NDArray[Float, 60, 60, 128], C4: NDArray[Float, 128, 128]):
    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))

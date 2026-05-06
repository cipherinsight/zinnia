# Source: NPBench go_fast (go_fast_numpy.py)
# Original signature: go_fast(a) — a is an (N, N) float array.
# Migration notes:
#   - N picked from the "S" preset (2000) shrunk to 16.
from zinnia import *

N = 16


@zk_circuit
def go_fast(a: NDArray[Float, 16, 16]):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    _zinnia_result = a + trace

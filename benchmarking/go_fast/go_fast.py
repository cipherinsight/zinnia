# Source: NPBench go_fast (go_fast_numpy.py)
# Original signature: go_fast(a) — a is an (N, N) float array.
# Migration notes:
#   - N picked from the "S" preset (2000).
from zinnia import *

N = 2000


@zk_circuit
def go_fast(a: NDArray[Float, 2000, 2000]):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    _zinnia_result = a + trace

# Source: NPBench deep_learning/softmax (softmax_numpy.py)
# Original signature: softmax(x) — x is an (N, H, SM, SM) float array.
# Migration notes:
#   - N, H, SM hoisted to module-level constants.
#   - From "S" preset (N=16, H=16, SM=128) shrunk to N=H=SM=8.
from zinnia import *

N = 8
H = 8
SM = 8


# Numerically-stable version of softmax
@zk_circuit
def softmax(x: NDArray[Float, 8, 8, 8, 8]):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    _zinnia_result = tmp_out / tmp_sum

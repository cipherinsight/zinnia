# Source: NPBench deep_learning/mlp (mlp_numpy.py)
# Original signature: mlp(input, w1, b1, w2, b2, w3, b3) — input (N, C_in); weights wi/biases bi.
# Migration notes:
#   - C_in, N, S0, S1, S2 hoisted to module-level constants.
#   - From "S" preset (C_in=3, N=8, S0=30000, S1=2000, S2=2000).
#   - Helpers (relu, softmax) kept as plain functions (no decorator).
from zinnia import *

C_IN = 3
N = 8
S0 = 30000
S1 = 2000
S2 = 2000


@zk_chip
def relu(x) -> NDArray[Float, 1]:
    return np.maximum(x, 0)


# Numerically-stable version of softmax
@zk_chip
def softmax(x) -> NDArray[Float, 1]:
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


# 3-layer MLP
@zk_circuit
def mlp(input: NDArray[Float, 8, 3],
        w1: NDArray[Float, 3, 30000], b1: NDArray[Float, 30000],
        w2: NDArray[Float, 30000, 2000], b2: NDArray[Float, 2000],
        w3: NDArray[Float, 2000, 2000], b3: NDArray[Float, 2000]):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)  # Softmax call can be omitted if necessary
    _zinnia_result = x

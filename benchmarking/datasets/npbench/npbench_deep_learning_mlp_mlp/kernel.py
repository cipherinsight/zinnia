# npbench program deep_learning/mlp/mlp_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_chip
def relu(x):
    return np.maximum(x, 0)

@zk_chip
def softmax(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum

@zk_circuit
def mlp(input: Float[Array, '8 3'], w1: Float[Array, '3 30000'], b1: Float[Array, '30000'], w2: Float[Array, '30000 2000'], b2: Float[Array, '2000'], w3: Float[Array, '2000 2000'], b3: Float[Array, '2000']):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)
    return x

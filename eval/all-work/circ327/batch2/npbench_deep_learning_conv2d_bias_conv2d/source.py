# npbench program deep_learning/conv2d_bias/conv2d_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_chip
def conv2d(input, weights):
    K = weights.shape[0]
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            output[:, i, j, :] = np.sum(input[:, i:i + K, j:j + K, :, np.newaxis] * weights[np.newaxis, :, :, :], axis=(1, 2, 3))
    return output

@zk_circuit
def conv2d_bias(input: Float[Array, '4 16 16 2'], weights: Float[Array, '2 2 2 8'], bias: Float[Array, '8']):
    return conv2d(input, weights) + bias

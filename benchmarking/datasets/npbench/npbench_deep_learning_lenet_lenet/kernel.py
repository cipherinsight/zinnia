# npbench program deep_learning/lenet/lenet_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_chip
def relu(x):
    return np.maximum(x, 0)

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

@zk_chip
def maxpool2d(x):
    output = np.empty([x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]], dtype=x.dtype)
    for i in range(x.shape[1] // 2):
        for j in range(x.shape[2] // 2):
            output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2, :], axis=(1, 2))
    return output

@zk_circuit
def lenet5(input: Float[Array, '4 28 28 1'], conv1: Float[Array, '5 5 1 6'], conv1bias: Float[Array, '6'], conv2: Float[Array, '5 5 6 16'], conv2bias: Float[Array, '16'], fc1w: Float[Array, '256 120'], fc1b: Float[Array, '120'], fc2w: Float[Array, '120 84'], fc2b: Float[Array, '84'], fc3w: Float[Array, '84 10'], fc3b: Float[Array, '10'], N: Integer, C_before_fc1: Integer):
    x = relu(conv2d(input, conv1) + conv1bias)
    x = maxpool2d(x)
    x = relu(conv2d(x, conv2) + conv2bias)
    x = maxpool2d(x)
    x = np.reshape(x, (N, C_before_fc1))
    x = relu(x @ fc1w + fc1b)
    x = relu(x @ fc2w + fc2b)
    return x @ fc3w + fc3b

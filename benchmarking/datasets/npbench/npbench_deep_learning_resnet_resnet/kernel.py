# npbench program deep_learning/resnet/resnet_numpy.py (spcl/npbench npbench/benchmarks).
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
def batchnorm2d(x, eps=1e-05):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(std + eps)

@zk_circuit
def resnet_basicblock(input: Float[Array, '8 14 14 32'], conv1: Float[Array, '1 1 32 8'], conv2: Float[Array, '3 3 8 8'], conv3: Float[Array, '1 1 8 32']):
    padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2, conv1.shape[3]))
    padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
    x = batchnorm2d(padded)
    x = relu(x)
    x = conv2d(x, conv2)
    x = batchnorm2d(x)
    x = relu(x)
    x = conv2d(x, conv3)
    x = batchnorm2d(x)
    return relu(x + input)

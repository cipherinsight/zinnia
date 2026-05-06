# Source: NPBench deep_learning/lenet (lenet_numpy.py)
# Original signature: lenet5(input, conv1, conv1bias, conv2, conv2bias,
#   fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, N, C_before_fc1).
# Migration notes:
#   - N, H, W hoisted to module-level constants; from "S" preset (N=4, H=W=28).
#   - Derived shapes (H_pool2, W_pool2, C_BEFORE_FC1) computed at module level mirroring initialize().
#   - Helpers (relu, conv2d, maxpool2d) kept as plain functions (no decorator).
from zinnia import *

N = 4
H = 28
W = 28

# Derived shapes mirroring initialize()
H_CONV1 = H - 4
W_CONV1 = W - 4
H_POOL1 = H_CONV1 // 2
W_POOL1 = W_CONV1 // 2
H_CONV2 = H_POOL1 - 4
W_CONV2 = W_POOL1 - 4
H_POOL2 = H_CONV2 // 2
W_POOL2 = W_CONV2 // 2
C_BEFORE_FC1 = 16 * H_POOL2 * W_POOL2


@zk_chip
def relu(x) -> NDArray[Float, 1]:
    return np.maximum(x, 0)


# Deep learning convolutional operator (stride = 1)
@zk_chip
def conv2d(input, weights) -> NDArray[Float, 1]:
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(H_out):
        for j in range(W_out):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] *
                weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


# 2x2 maxpool operator, as used in LeNet-5
@zk_chip
def maxpool2d(x) -> NDArray[Float, 1]:
    output = np.empty(
        [x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]],
        dtype=x.dtype)
    for i in range(x.shape[1] // 2):
        for j in range(x.shape[2] // 2):
            output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2,
                                          2 * j:2 * j + 2, :],
                                        axis=(1, 2))
    return output


# LeNet-5 Convolutional Neural Network (inference mode)
@zk_circuit
def lenet(input: NDArray[Float, 4, 28, 28, 1],
          conv1: NDArray[Float, 5, 5, 1, 6],
          conv1bias: NDArray[Float, 6],
          conv2: NDArray[Float, 5, 5, 6, 16],
          conv2bias: NDArray[Float, 16],
          fc1w: NDArray[Float, 256, 120],
          fc1b: NDArray[Float, 120],
          fc2w: NDArray[Float, 120, 84],
          fc2b: NDArray[Float, 84],
          fc3w: NDArray[Float, 84, 10],
          fc3b: NDArray[Float, 10]):
    x = relu(conv2d(input, conv1) + conv1bias)
    x = maxpool2d(x)
    x = relu(conv2d(x, conv2) + conv2bias)
    x = maxpool2d(x)
    x = np.reshape(x, (4, C_BEFORE_FC1))
    x = relu(x @ fc1w + fc1b)
    x = relu(x @ fc2w + fc2b)
    _zinnia_result = x @ fc3w + fc3b

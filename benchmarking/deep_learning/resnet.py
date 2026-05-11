# Source: NPBench deep_learning/resnet (resnet_numpy.py)
# Original signature: resnet_basicblock(input, conv1, conv2, conv3) —
#   input (N, H, W, C1); conv1 (1,1,C1,C2); conv2 (3,3,C2,C2); conv3 (1,1,C2,C1).
# Migration notes:
#   - N, H, W, C1, C2 hoisted to module-level constants.
#   - From "S" preset (N=8, W=H=14, C1=32, C2=8).
#   - Helpers (relu, conv2d, batchnorm2d) kept as plain functions (no decorator).
from zinnia import *

N = 8
H = 14
W = 14
C1 = 32
C2 = 8


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


# Batch normalization operator, as used in ResNet
@zk_chip
def batchnorm2d(x, eps=1e-5) -> NDArray[Float, 1]:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(std + eps)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
@zk_circuit
def resnet(input: NDArray[Float, 8, 14, 14, 32],
           conv1: NDArray[Float, 1, 1, 32, 8],
           conv2: NDArray[Float, 3, 3, 8, 8],
           conv3: NDArray[Float, 1, 1, 8, 32]):
    # Pad output of first convolution for second convolution
    padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
                       conv1.shape[3]))

    padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
    x = batchnorm2d(padded)
    x = relu(x)

    x = conv2d(x, conv2)
    x = batchnorm2d(x)
    x = relu(x)
    x = conv2d(x, conv3)
    x = batchnorm2d(x)
    _zinnia_result = relu(x + input)

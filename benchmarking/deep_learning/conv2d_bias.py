# Source: NPBench deep_learning/conv2d_bias (conv2d_numpy.py)
# Original signature: conv2d_bias(input, weights, bias) —
#   input (N, H, W, C_in), weights (K, K, C_in, C_out), bias (C_out,) float arrays.
# Migration notes:
#   - All shape params (N, H, W, K, C_in, C_out) hoisted to module-level constants.
#   - From "S" preset (N=8, C_in=3, C_out=16, K=2, H=W=32) shrunk to H=W=16, C_out=8 (others fit).
#   - Helper conv2d kept as a plain function (no decorator).
from zinnia import *

N = 8
H = 16
W = 16
K = 2
C_IN = 3
C_OUT = 8


# Deep learning convolutional operator (stride = 1)
@zk_chip
def conv2d(input, weights) -> NDArray[Float, 8, 15, 15, 8]:
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


@zk_circuit
def conv2d_bias(input: NDArray[Float, 8, 16, 16, 3],
                weights: NDArray[Float, 2, 2, 3, 8],
                bias: NDArray[Float, 8]):
    _zinnia_result = conv2d(input, weights) + bias

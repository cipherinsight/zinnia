# Source: Pythran tests/cases/convnet.py
# Original #pythran export: convnet(float32[:,:], float32[:,:], int, int, int)
from zinnia import *

M = 4
N = 4


@zk_chip
def sigmoid(z) -> Float:
    return 1 / (1 + np.exp(-z))


@zk_circuit
def convnet(conv_matrix: NDArray[Float, 4, 4], qcnn_filter: NDArray[Float, 4, 4],
            length: int, batch_size: int, state_size: int):
    state_size_2 = state_size * 2
    state_size_3 = state_size * 3
    state_size_4 = state_size * 4

    conv_results = np.dot(conv_matrix, qcnn_filter).reshape(length, batch_size, state_size_4)
    conv_results[:, :, :state_size] = np.tanh(conv_results[:, :, :state_size])
    conv_results[:, :, state_size:state_size_4] = sigmoid(conv_results[:, :, state_size:state_size_4])

    state_results = np.zeros((batch_size, length, state_size_2), dtype=np.float32)
    state = np.zeros((batch_size, state_size), dtype=np.float32)
    for i in range(length):
        z = conv_results[i, :, :state_size]
        f = conv_results[i, :, state_size:state_size_2]
        o = conv_results[i, :, state_size_2:state_size_3]
        state = f * state + (1 - f) * z
        state_results[:, i, :state_size] = state * o

    state = np.zeros((batch_size, state_size), dtype=np.float32)
    for i in range(length - 1, -1, -1):
        z = conv_results[i, :, :state_size]
        f = conv_results[i, :, state_size_3:state_size_4]
        o = conv_results[i, :, state_size_2:state_size_3]
        state = f * state + (1 - f) * z
        state_results[:, i, state_size:] = state * o
    state_results = state_results.reshape((batch_size * length, state_size_2))
    _zinnia_result = state_results

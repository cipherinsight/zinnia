from zinnia import *
import numpy as np

@zk_chip
def relu(x: Float[Array, 'N<=1024']):
    return np.maximum(x, 0)

@zk_chip
def conv2d(input: Float[Array, 'N<=1024 M<=1024 K<=1024 L<=256'], weights: Float[Array, 'N<=1024 M<=1024 K<=1024']):
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
def maxpool2d(x: Float[Array, 'N<=1024 M<=1024 K<=1024 L<=256']):
    output = np.empty([x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]], dtype=x.dtype)
    for i in range(x.shape[1] // 2):
        for j in range(x.shape[2] // 2):
            output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2, :], axis=(1, 2))
    return output

@zk_circuit
def lenet5(input: Float[Array, 'N<=1024 H<=1024 W<=1024 C<=256'], conv1: Float[Array, 'K<=256 K<=256 C<=256 F1<=256'], conv1bias: Float[Array, 'F1<=256'], conv2: Float[Array, 'K2<=256 K2<=256 F1<=256 F2<=256'], conv2bias: Float[Array, 'F2<=256'], fc1w: Float[Array, 'C0<=1024 C1<=1024'], fc1b: Float[Array, 'C1<=1024'], fc2w: Float[Array, 'C1<=1024 C2<=1024'], fc2b: Float[Array, 'C2<=1024'], fc3w: Float[Array, 'C2<=1024 C3<=1024'], fc3b: Float[Array, 'C3<=1024'], N: Integer, C_before_fc1: Integer):
    x = relu(conv2d(input, conv1) + conv1bias)
    x = maxpool2d(x)
    x = relu(conv2d(x, conv2) + conv2bias)
    x = maxpool2d(x)
    x = np.reshape(x, (N, C_before_fc1))
    x = relu(x @ fc1w + fc1b)
    x = relu(x @ fc2w + fc2b)
    return x @ fc3w + fc3b

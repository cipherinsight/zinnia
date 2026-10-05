from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, domain-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
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
def batchnorm2d(x: Float[Array, 'N<=1024'], eps: Float=1e-05):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(std + eps)

@zk_circuit
def resnet_basicblock(input: Float[Array, 'N<=1024 H<=1024 W<=1024 C<=256'], conv1: Float[Array, 'K1<=256 K1<=256 C<=256 F1<=256'], conv2: Float[Array, 'K2<=256 K2<=256 F1<=256 F2<=256'], conv3: Float[Array, 'K3<=256 K3<=256 F2<=256 C<=256']):
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

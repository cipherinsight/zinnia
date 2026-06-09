from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_chip
def conv2d(input: Float[Array, 'N<=1024 H<=1024 W<=1024 C<=256'], weights: Float[Array, 'K<=256 K<=256 C<=256 F<=256']):
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
def conv2d_bias(input: Float[Array, 'N<=1024 H<=1024 W<=1024 C<=256'], weights: Float[Array, 'K<=256 K<=256 C<=256 F<=256'], bias: Float[Array, 'F<=256']):
    return conv2d(input, weights) + bias

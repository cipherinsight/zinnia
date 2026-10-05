from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded, matmul-dot
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Declare shared matrix/vector dimensions explicitly; when not expressible as @requires, encode with matching symbolic annotation names and keep as review obligation.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_chip
def sigmoid(z: Float[Array, 'N<=1024']):
    return 1 / (1 + np.exp(-z))

@zk_circuit
def convnet(conv_matrix: Float[Array, 'N<=1024 M<=1024'], qcnn_filter: Float[Array, 'N<=1024 M<=1024'], length: Integer, batch_size: Integer, state_size: Integer):
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
    return state_results

from zinnia import *
import numpy as np

@zk_chip
def relu(x: Float[Array, 'N<=1024']):
    return np.maximum(x, 0)

@zk_chip
def softmax(x: Float[Array, 'N<=1024']):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum

@zk_circuit
def mlp(input: Float[Array, 'N<=1024'], w1: Float[Array, 'N<=1024 H<=1024'], b1: Float, w2: Float[Array, 'H<=1024 K<=1024'], b2: Float, w3: Float[Array, 'K<=1024 M<=1024'], b3: Float):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)
    return x

from zinnia import *
import numpy as np

@zk_circuit
def kernel(alpha: Float, beta: Float, A: Float[Array, 'N<=1024 N<=1024'], u1: Float[Array, 'N<=1024'], v1: Float[Array, 'N<=1024'], u2: Float[Array, 'N<=1024'], v2: Float[Array, 'N<=1024'], w: Float, x: Float[Array, 'N<=1024'], y: Float[Array, 'N<=1024'], z: Float):
    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x

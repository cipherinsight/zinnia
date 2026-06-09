from zinnia import *
import numpy as np

@zk_circuit
def kernel(alpha: Float, beta: Float, C: Float[Array, 'N<=1024 M<=1024'], A: Float[Array, 'N<=1024 K<=1024'], B: Float[Array, 'K<=1024 M<=1024']):
    C[:] = alpha * A @ B + beta * C

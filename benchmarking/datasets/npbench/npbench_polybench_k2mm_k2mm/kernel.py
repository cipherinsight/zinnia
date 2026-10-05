from zinnia import *
import numpy as np

@zk_circuit
def kernel(alpha: Float, beta: Float, A: Float[Array, 'N<=1024 K<=1024'], B: Float[Array, 'K<=1024 M<=1024'], C: Float[Array, 'M<=1024 L<=256'], D: Float[Array, 'N<=1024 L<=256']):
    D[:] = alpha * A @ B @ C + beta * D

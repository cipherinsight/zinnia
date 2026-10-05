from zinnia import *
import numpy as np

@zk_circuit
def kernel(A: Float[Array, 'N<=1024 K<=1024'], B: Float[Array, 'K<=1024 M<=1024'], C: Float[Array, 'M<=1024 L<=256'], D: Float[Array, 'L<=256 R<=256']):
    return A @ B @ C @ D

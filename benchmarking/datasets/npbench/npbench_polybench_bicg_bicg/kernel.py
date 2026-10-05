from zinnia import *
import numpy as np

@zk_circuit
def kernel(A: Float[Array, 'N<=1024 M<=1024'], p: Float[Array, 'M<=1024'], r: Float[Array, 'N<=1024']):
    return (r @ A, A @ p)

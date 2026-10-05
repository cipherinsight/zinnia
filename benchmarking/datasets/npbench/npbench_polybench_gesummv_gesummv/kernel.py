from zinnia import *
import numpy as np

@zk_circuit
def kernel(alpha: Float, beta: Float, A: Float[Array, 'N<=1024 N<=1024'], B: Float[Array, 'N<=1024 N<=1024'], x: Float[Array, 'N<=1024']):
    return alpha * A @ x + beta * B @ x

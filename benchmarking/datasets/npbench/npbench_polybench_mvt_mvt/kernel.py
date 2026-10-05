from zinnia import *
import numpy as np

@zk_circuit
def kernel(x1: Float, x2: Float, y_1: Float[Array, 'M<=1024'], y_2: Float[Array, 'N<=1024'], A: Float[Array, 'N<=1024 M<=1024']):
    x1 += A @ y_1
    x2 += y_2 @ A

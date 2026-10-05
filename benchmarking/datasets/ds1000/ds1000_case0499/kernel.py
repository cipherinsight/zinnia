from zinnia import *
import numpy as np

# DS-1000 problem 499 (NumPy problem 208; Difficult-Rewrite perturbation of problem 495).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 4, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: shape is a compile-time constant (test case 4 value).

@zk_chip
def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                        (x_pad//2, x_pad//2 + x_pad%2)),
                    mode = 'constant')

@zk_circuit
def kernel(a: Float[Array, "42 10"]):
    shape = (93, 13)
    result = to_shape(a, shape)
    return result

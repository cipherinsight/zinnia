from zinnia import *
import numpy as np

# DS-1000 problem 390 (NumPy problem 99; Difficult-Rewrite perturbation of problem 385).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: patch_size is a compile-time constant (test case 2 value).

@zk_circuit
def kernel(a: Float[Array, "100 200"]):
    patch_size = 5
    x = a[:a.shape[0] // patch_size * patch_size, :a.shape[1] // patch_size * patch_size]
    result = x.reshape(x.shape[0]//patch_size, patch_size, x.shape[1]// patch_size, patch_size).swapaxes(1, 2).transpose(1, 0, 2, 3).reshape(-1, patch_size, patch_size)
    return result

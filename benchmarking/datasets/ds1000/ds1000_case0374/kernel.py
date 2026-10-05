from zinnia import *
import numpy as np

# DS-1000 problem 374 (NumPy problem 83; Difficult-Rewrite perturbation of problem 373).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_chip
def ecdf_result(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

@zk_circuit
def kernel(grades: Float[Array, "50"], eval: Integer[Array, "5"]):
    resultx, resulty = ecdf_result(grades)
    result = np.zeros_like(eval, dtype=float)
    for i, element in enumerate(eval):
        if element < resultx[0]:
            result[i] = 0
        elif element >= resultx[-1]:
            result[i] = 1
        else:
            result[i] = resulty[(resultx > element).argmax()-1]
    return result

from zinnia import *
import numpy as np

# DS-1000 problem 375 (NumPy problem 84; Difficult-Rewrite perturbation of problem 373).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_chip
def ecdf_result(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

@zk_circuit
def kernel(grades: Float[Array, "50"], threshold: Float):
    resultx, resulty = ecdf_result(grades)
    t = (resulty > threshold).argmax()
    low = resultx[0]
    high = resultx[t]
    return [low, high]

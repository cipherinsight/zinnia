from zinnia import *
import numpy as np

# DS-1000 problem 490 (NumPy problem 199; Semantic perturbation of problem 489).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(arr: Float[Array, "5 50"], n1: Integer[Array, "5"], n2: Integer[Array, "5"]):
    for a, t1, t2 in zip(arr, n1, n2):
        temp = a.copy()
        a[np.where(temp < t1)] = 0
        a[np.where(temp >= t2)] = 30
        a[np.logical_and(temp >= t1, temp < t2)] += 5
    return arr

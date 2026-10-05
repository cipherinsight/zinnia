from zinnia import *
import numpy as np

# DS-1000 problem 429 (NumPy problem 138; Semantic perturbation of problem 428).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "30"]):
    result = (a.mean()-2*a.std(), a.mean()+2*a.std())
    return result

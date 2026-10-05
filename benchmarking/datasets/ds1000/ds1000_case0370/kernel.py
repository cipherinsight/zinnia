from zinnia import *
import numpy as np

# DS-1000 problem 370 (NumPy problem 79; Surface perturbation of problem 368).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 5, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "3 3"]):
    result = np.isclose(a, a[0], atol=0).all()

    return result

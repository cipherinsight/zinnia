from zinnia import *
import numpy as np

# DS-1000 problem 318 (NumPy problem 27; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
@requires(lambda a, permutation: is_permutation(permutation))
def kernel(a: Float[Array, "8 6"], permutation: Integer[Array, "6"]):
    c = np.empty_like(permutation)
    c[permutation] = np.arange(len(permutation))
    a = a[:, c]
    return a

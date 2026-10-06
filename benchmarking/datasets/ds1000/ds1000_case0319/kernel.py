from zinnia import *
import numpy as np

# DS-1000 problem 319 (NumPy problem 28; Semantic perturbation of problem 318).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
@requires(lambda a, permutation: is_permutation(permutation))
def kernel(a: Float[Array, "8 6 8"], permutation: Integer[Array, "8"]):
    c = np.empty_like(permutation)
    c[permutation] = np.arange(len(permutation))
    result = a[c, :, :]
    return result

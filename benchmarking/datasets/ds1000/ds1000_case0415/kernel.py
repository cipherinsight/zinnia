from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(data: Float[Array, "8"], bin_size: Integer):
    bin_data_max = data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size).max(axis=1)
    return bin_data_max

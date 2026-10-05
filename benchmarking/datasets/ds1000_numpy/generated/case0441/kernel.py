from zinnia import *
import numpy as np

# DS-1000 problem 441 (NumPy problem 150; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Integer[Array, "4"], number: Integer):
    is_contained = number in a
    return is_contained

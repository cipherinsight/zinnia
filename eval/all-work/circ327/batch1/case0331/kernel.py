from zinnia import *
import numpy as np

# DS-1000 problem 331 (NumPy problem 40; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(numerator: Integer, denominator: Integer):
    gcd = np.gcd(numerator, denominator)
    result = (numerator//gcd, denominator//gcd)
    return result

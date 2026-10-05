from zinnia import *
import numpy as np

# DS-1000 problem 354 (NumPy problem 63; Semantic perturbation of problem 353).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(A: Integer[Array, "2 3"], B: Integer[Array, "7 3"]):
    dims = np.maximum(B.max(0),A.max(0))+1
    result = A[~np.in1d(np.ravel_multi_index(A.T,dims),np.ravel_multi_index(B.T,dims))]
    output = np.append(result, B[~np.in1d(np.ravel_multi_index(B.T,dims),np.ravel_multi_index(A.T,dims))], axis = 0)
    return output

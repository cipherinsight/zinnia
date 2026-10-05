from zinnia import *
import numpy as np

# DS-1000 problem 348 (NumPy problem 57; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_chip
def LI_vecs(M):
    dim = M.shape[0]
    LI=[M[0]]
    for i in range(dim):
        tmp=[]
        for r in LI:
            tmp.append(r)
        tmp.append(M[i])                #set tmp=LI+[M[i]]
        if np.linalg.matrix_rank(tmp)>len(LI):    #test if M[i] is linearly independent from all (row) vectors in LI
            LI.append(M[i])             #note that matrix_rank does not need to take in a square matrix
    return LI                           #return set of linearly independent (row) vectors

@zk_circuit
def kernel(a: Integer[Array, "5 6"]):
    result = LI_vecs(a)
    return result

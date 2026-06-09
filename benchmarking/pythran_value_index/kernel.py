from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, sort-permutation
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: Sort/permutation kernels should remain explicit diagnostic rows until supported.
import numpy as np

@zk_circuit
def tri(score: Integer[Array, 'N<=1024 M<=1024'], score_t: Integer[Array, 'N<=1024 M<=1024']):
    score_t[0, 0] = score[0, 0]
    score_t = score[np.argsort(score[:, 1])]
    return score_t

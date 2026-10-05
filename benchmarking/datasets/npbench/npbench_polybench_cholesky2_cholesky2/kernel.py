from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, linalg
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Record square/invertible/SPD requirements where the linalg call needs them.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: np.linalg kernels may stay fail-closed until support is selected.
import numpy as np

@zk_circuit
def kernel(A: Float[Array, 'N<=1024 N<=1024']):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)

from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded, matmul-dot
# Expected outcome: compile/prove candidate
# Precondition: Declare shared matrix/vector dimensions explicitly; when not expressible as @requires, encode with matching symbolic annotation names and keep as review obligation.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def kernel(alpha: Float, A: Float[Array, 'N<=1024 M<=1024'], B: Float[Array, 'N<=1024 M<=1024']):
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha

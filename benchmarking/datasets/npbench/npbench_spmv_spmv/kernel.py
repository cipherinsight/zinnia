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
def spmv(A_row: Integer[Array, 'N<=1024'], A_col: Integer[Array, 'N<=1024'], A_val: Float[Array, 'N<=1024'], x: Float[Array, 'N<=1024']):
    y = np.empty(A_row.size - 1, A_val.dtype)
    for i in range(A_row.size - 1):
        cols = A_col[A_row[i]:A_row[i + 1]]
        vals = A_val[A_row[i]:A_row[i + 1]]
        y[i] = vals @ x[cols]
    return y

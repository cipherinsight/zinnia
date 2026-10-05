from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.

@zk_chip
def zero(n: Integer, m: Integer):
    return [[0 for row in range(n)] for col in range(m)]

@zk_circuit
def matrix_multiply(m0: Float[Array, 'N<=1024 M<=1024'], m1: Float[Array, 'N<=1024 M<=1024']):
    new_matrix = zero(len(m0), len(m1[0]))
    for i in range(len(m0)):
        for j in range(len(m1[0])):
            r = 0
            'omp parallel for reduction(+:r)'
            for k in range(len(m1)):
                r += m0[i][k] * m1[k][j]
            new_matrix[i][j] = r
    return new_matrix

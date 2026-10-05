from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def factorMatrix0(M: Float[Array, 'N<=1024 M<=1024']):
    n = M.shape[0]
    m = M.shape[1]
    for line in range(0, n - 1):
        cmax = np.argmax(abs(M[line:n, line])) + line
        if cmax != line:
            M[[line, cmax]] = M[[cmax, line]]
        pivot = M[line, line]
        for j in range(line + 1, n):
            v = M[j, line] / pivot
            for k in range(line, m):
                M[j, k] -= v * M[line, k]

@zk_circuit
def factorMatrix1(M: Float[Array, 'N<=1024 M<=1024']):
    n = M.shape[0]
    m = M.shape[1]
    for line in range(0, n - 1):
        cmax = line
        vmax = abs(M[line, line])
        for i in range(line + 1, n):
            if abs(M[i, line]) > vmax:
                vmax = abs(M[i, line])
                cmax = i
        if cmax != line:
            for j in range(line, m):
                t = M[line, j]
                M[line, j] = M[cmax, j]
                M[cmax, j] = t
        pivot = M[line, line]
        for j in range(line + 1, n):
            v = M[j, line] / pivot
            for k in range(line, m):
                M[j, k] -= v * M[line, k]

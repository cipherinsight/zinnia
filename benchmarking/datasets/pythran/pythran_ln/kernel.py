from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.

@zk_circuit
def ln(X: Float[Array, 'N<=1024'], Y: Float[Array, 'N<=1024']):
    Y[:] = X - 1 - (X - 1) ** 2 / 2 + (X - 1) ** 3 / 3 - (X - 1) ** 4 / 4 + (X - 1) ** 5 / 5 - (X - 1) ** 6 / 6 + (X - 1) ** 7 / 7 - (X - 1) ** 8 / 8 + (X - 1) ** 9 / 9
    return Y

from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation
# Expected outcome: compile/prove candidate
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.

@zk_circuit
def periodic_bc(f: Float[Array, 'N<=1024 M<=1024 K<=1024']):
    f[:, 0, :] = f[:, -2, :]
    return f

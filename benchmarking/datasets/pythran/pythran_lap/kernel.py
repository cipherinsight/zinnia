from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy

@zk_circuit
@requires(lambda var, dh2: dh2 != 0.0)
def laplacian(var: Float[Array, 'N<=1024'], dh2: Float):
    """ (1D array, dx^2) -> laplacian(1D array)
    periodic_laplacian_1D_4th_order
    Implementing the 4th order 1D laplacian with periodic condition
    """
    lap = numpy.zeros_like(var)
    lap[1:] = 4.0 / 3.0 * var[:-1]
    lap[0] = 4.0 / 3.0 * var[1]
    lap[:-1] += 4.0 / 3.0 * var[1:]
    lap[-1] += 4.0 / 3.0 * var[0]
    lap += -5.0 / 2.0 * var
    lap[2:] += -1.0 / 12.0 * var[:-2]
    lap[:2] += -1.0 / 12.0 * var[-2:]
    lap[:-2] += -1.0 / 12.0 * var[2:]
    lap[-2:] += -1.0 / 12.0 * var[:2]
    return lap / dh2

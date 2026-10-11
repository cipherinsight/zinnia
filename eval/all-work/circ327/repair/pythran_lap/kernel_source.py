# pythran program lap.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy

@zk_circuit
@requires(lambda var, dh2: dh2 != 0.0)
def laplacian(var: Float[Array, '16'], dh2: Float):
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

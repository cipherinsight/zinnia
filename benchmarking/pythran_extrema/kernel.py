from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: shape-dtype-only
# Expected outcome: compile/prove candidate
# Precondition: Shape/rank/dtype annotations only.
from functools import reduce

@zk_circuit
def extrema_op(a: Tuple[Integer, Float, Integer, Float], b: Tuple[Integer, Float, Integer, Float]):
    a_min_idx, a_min_val, a_max_idx, a_max_val = a
    b_min_idx, b_min_val, b_max_idx, b_max_val = b
    if a_min_val < b_min_val:
        if a_max_val > b_max_val:
            return a
        else:
            return (a_min_idx, a_min_val, b_max_idx, b_max_val)
    elif a_max_val > b_max_val:
        return (b_min_idx, b_min_val, a_max_idx, a_max_val)
    else:
        return b

@zk_chip
def extrema_id(x: Float):
    return (-1, 1.0, 1, 0.0)

@zk_chip
def indices(A: Float[Array, 'N<=1024']):
    return range(len(A))

@zk_chip
def extrema(x: Float[Array, 'N<=1024'], x_id: Tuple[Integer, Float, Integer, Float]):
    return reduce(extrema_op, zip(indices(x), x, indices(x), x), x_id)

@zk_circuit
@requires(lambda n, a: 0 <= n <= 1024)
def run_extrema(n: Integer, a: Float[Array, 'N<=1024']):
    a_id = extrema_id(0.0)
    return extrema(a, a_id)

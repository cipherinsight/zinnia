# Source: Pythran tests/cases/extrema.py
# Original #pythran export: run_extrema(int, float list)
from zinnia import *
from functools import reduce


@zk_chip
def extrema_op(a, b) -> Tuple[Integer, Float, Integer, Float]:
    a_min_idx, a_min_val, a_max_idx, a_max_val = a
    b_min_idx, b_min_val, b_max_idx, b_max_val = b
    if a_min_val < b_min_val:
        if a_max_val > b_max_val:
            return a
        else:
            return a_min_idx, a_min_val, b_max_idx, b_max_val
    else:
        if a_max_val > b_max_val:
            return b_min_idx, b_min_val, a_max_idx, a_max_val
        else:
            return b


@zk_chip
def extrema_id(x) -> Tuple[Integer, Float, Integer, Float]:
    return -1, 1., 1, 0.


@zk_chip
def indices(A) -> List[Integer]:
    return range(len(A))


@zk_chip
def extrema(x, x_id) -> Tuple[Integer, Float, Integer, Float]:
    return reduce(extrema_op, zip(indices(x), x, indices(x), x), x_id)


@zk_circuit
def run_extrema(n: int, a: NDArray[Float, 64]):
    a_id = extrema_id(0.)
    _zinnia_result = extrema(a, a_id)

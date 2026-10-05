# pythran program extrema.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
from functools import reduce

@zk_chip
def extrema_op(a, b):
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
def extrema_id(x):
    return (-1, 1.0, 1, 0.0)

@zk_chip
def indices(A):
    return range(len(A))

@zk_chip
def extrema(x, x_id):
    return reduce(extrema_op, zip(indices(x), x, indices(x), x), x_id)

@zk_circuit
@requires(lambda n, a: 0 <= n <= 1024)
def run_extrema(n: Integer, a: Float[Array, '10']):
    a_id = extrema_id(0.0)
    return extrema(a, a_id)

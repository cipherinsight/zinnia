# npbench program spmv/spmv_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def spmv(A_row: Integer[Array, '9'], A_col: Integer[Array, '16'], A_val: Float[Array, '16'], x: Float[Array, '8']):
    y = np.empty(A_row.size - 1, A_val.dtype)
    for i in range(A_row.size - 1):
        cols = A_col[A_row[i]:A_row[i + 1]]
        vals = A_val[A_row[i]:A_row[i + 1]]
        y[i] = vals @ x[cols]
    return y

# Source: NPBench spmv (spmv_numpy.py)
# Original signature: spmv(A_row, A_col, A_val, x) — CSR sparse matvec.
#   A_row: (M+1,) int, A_col: (NNZ,) int, A_val: (NNZ,) float, x: (N,) float.
# Migration notes:
#   - M, N, NNZ hoisted to module-level constants. From "S" preset (M=N=4096, nnz=8192) shrunk to M=N=16, NNZ=16.
#   - Data-dependent slicing A_col[A_row[i]:A_row[i+1]] is likely unsupported (dynamic slice bounds).
from zinnia import *

M = 16
N = 16
NNZ = 16


@zk_circuit
def spmv(A_row: NDArray[Integer, 17], A_col: NDArray[Integer, 16],
         A_val: NDArray[Float, 16], x: NDArray[Float, 16]):
    y = np.empty(A_row.size - 1, A_val.dtype)

    for i in range(A_row.size - 1):
        cols = A_col[A_row[i]:A_row[i + 1]]
        vals = A_val[A_row[i]:A_row[i + 1]]
        y[i] = vals @ x[cols]

    _zinnia_result = y

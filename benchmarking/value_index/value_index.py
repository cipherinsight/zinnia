# Source: Pythran tests/cases/value_index.py
# Original #pythran export: tri(int[][], int[][])
from zinnia import *

M = 16
N = 16


@zk_circuit
def tri(score: NDArray[Integer, 16, 16], score_t: NDArray[Integer, 16, 16]):
    score_t[0, 0] = score[0, 0]
    score_t = score[np.argsort(score[:, 1])]
    _zinnia_result = score_t

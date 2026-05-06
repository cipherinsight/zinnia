# Source: Pythran tests/cases/pythagorean_triples.py
# Original #pythran export: next_pythagorean_triples(int64[:,:])
from zinnia import *

M = 4
N = 3


@zk_circuit
def next_pythagorean_triples(previous: NDArray[Integer, 4, 3]):
    matrices = np.array(
        [[-1, 2, 2],
         [-2, 1, 2],
         [-2, 2, 3],
         [1, 2, 2],
         [2, 1, 2],
         [2, 2, 3],
         [1, -2, 2],
         [2, -1, 2],
         [2, -2, 3]])

    next_triples = np.transpose(np.dot(matrices, np.transpose(previous)))
    next_triples = next_triples.reshape((3 * previous.shape[0], previous.shape[1]))
    _zinnia_result = next_triples

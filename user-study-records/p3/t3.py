from zinnia import *
import numpy as np


@zk_chip
def is_path_exists(A: NDArray[int, 10, 10], a: int, b: int) -> int:
    reach = np.zeros((10, 10), dtype=int)

    for i in range(10):
        for j in range(10):
            reach[i, j] = A[i, j]
            # A node can reach itself
            if i == j:
                reach[i, j] = 1

    for k in range(10):
        for i in range(10):
            for j in range(10):
                # If i can reach k and k can reach j, then i can reach j
                if reach[i, k] == 1 and reach[k, j] == 1:
                    reach[i, j] = 1

    return reach[a, b]


@zk_circuit
def graph_connectivity(A: Public[NDArray[int, 10, 10]], a: Public[int], b: Public[int], y: int):
    assert 0 <= a and a <= 9
    assert 0 <= b and b <= 9

    assert y == 0 or y == 1

    for i in range(10):
        for j in range(10):
            assert A[i, j] == 0 or A[i, j] == 1

    correct_result = is_path_exists(A, a, b)

    assert y == correct_result

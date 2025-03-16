from zinnia import zk_circuit, NDArray


@zk_circuit
def test(A: NDArray[int, 10, 10], a: int, b: int, res: int):
    assert 0 <= a <= 9
    assert 0 <= b <= 9
    for k in range(10):
        for i in range(10):
            for j in range(10):
                A[i,j] = 1 if A[i,j] == 1 or (A[i,k] == 1 and A[k,j] == 1) else 0
    assert A[a,b] == res


matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

test(matrix, 1, 2, 1)
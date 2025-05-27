from zinnia import zk_circuit, NDArray


@zk_circuit
def matrix_dot_prod(mat_a: NDArray[int, 4, 4], mat_b: NDArray[int, 4, 4], result: NDArray[int, 4, 4]):
    assert mat_a @ mat_b == result


assert matrix_dot_prod(
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
    [[90, 100, 110, 120], [202, 228, 254, 280], [314, 356, 398, 440], [426, 484, 542, 600]]
)

from zinnia import zk_circuit, NDArray


@zk_circuit
def sum_over_array(ary: NDArray[int, 10], result: int):
    s = 0
    for i in range(10):
        s += ary[i]
    assert s == result


assert sum_over_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 55)
assert sum_over_array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 10)

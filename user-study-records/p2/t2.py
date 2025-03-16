from zinnia import zk_circuit, NDArray


@zk_circuit

def climb_stairs(n: int, y: int):
    assert n <= 50
    assert n >= 1
    if n <= 2:
        assert y == n
    else:
        a, b = 1, 2
        for i in range(3, 50 + 1):
            a, b = b, a + b
            if i == n:
                assert y == b

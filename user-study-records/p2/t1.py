from zinnia import zk_circuit, NDArray


@zk_circuit
def task1(num: int, y : int):
    assert num >= 1
    assert num <= 10000
    divisor = 2
    if num <= 1:
        # y should be 0
        assert y == 0
    elif divisor * divisor > num:
        # y should be 1
        assert y == 1
    else:
        yy = 1
        for i in range(2, 100+1):
            if i < num and num % i == 0:
                yy = 0
        assert y == yy

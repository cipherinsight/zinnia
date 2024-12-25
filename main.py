from pyzk.lang.typing import Public, Private
from pyzk.lang.type import Integer, Float, NDArray
from pyzk.pyzk_interface import pyzk_circuit, pyzk_chip


@pyzk_chip
def multi(a: Integer) -> NDArray[Float, 2, 3]:
    if a == 0:
        return NDArray.ones((2, 1), Integer) @ NDArray.ones((1, 3))
    else:
        if a == 1:
            return NDArray.ones((2, 3), Float)
        else:
            return NDArray.zeros((2, 3))


@pyzk_chip
def fibonacci(a: Integer) -> Integer:
    if a == 1:
        return 1
    else:
        if a == 2:
            return 1
        else:
            return fibonacci(a - 1) + fibonacci(a - 2)


@pyzk_chip
def is_thirteen(x: Integer):
    assert x == 13


@pyzk_circuit
def foo(
    x: Public[Integer],
    y: Private[NDArray[Integer, 5, 5]],
):
    for i in range(12):
        y = y @ (y + 1)
    assert y.sum() != 0


foo(7, [
    [1, 2, 3, 4, -5],
    [-5, 213, 13, 14, 123],
    [65, 64, -24, 52, -12],
    [72, 7, -4, 25, 44],
    [3, 24, 1, 94, -123]
])

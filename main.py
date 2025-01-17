import math

from pyzk import sin, cos, concatenate, stack, tan
from pyzk.lang.typing import Public, Private
from pyzk.lang.type import Integer, Float, NDArray
from pyzk.pyzk_interface import pyzk_circuit, pyzk_chip

@pyzk_chip
def fibonacci(a: Integer) -> Integer:
    assert a >= 0
    if a == 0:
        return 0
    if a == 1:
        return 1
    return fibonacci(a - 1) + fibonacci(a - 2)


@pyzk_circuit
def main(
    A: Public[NDArray[Float, 4, 4]],
    B: Public[NDArray[Float, 4, 4]],
    x: Private[Integer]
):
    C = NDArray.eye(4, 4)
    for i in range(4):
        C[i, i] = float(i + 1)
    assert all(A @ B == C)
    sliced = A[1:3, 1:3]
    the_prod = 1
    for i in range(1, 10):
        if i == (sliced).sum():
            break
        the_prod *= i
    assert the_prod == x
    # assert fibonacci(2) == x


main([1, 2, 3, 4, 5, 6, 7, 13], 7)

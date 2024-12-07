from pyzk.lang.typing import Public, Private
from pyzk.lang.type import Number, NDArray
from pyzk.pyzk_interface import pyzk_circuit, pyzk_chip


@pyzk_chip
def another_chip(a: Number, b: Number) -> Number:
    if a != 0:
        return another_chip(0, b)
    return a + 123 * b


@pyzk_chip
def my_chip(a: Number) -> Number:
    if a == 0:
        return 1
    else:
        if a == 1:
            return 2
        else:
            return another_chip(1, a)


@pyzk_chip
def returns_none(a: Number):
    assert a != 0


@pyzk_circuit
def foo(
    x: Public[Number],
    y: Private[NDArray[5, 4]]
):
    z = y @ y.transpose(axes=(1, 0))
    assert my_chip(x) == 0
    returns_none(x)
    assert (z + NDArray.ones((5, 5))).sum() == 0


foo()



from pyzk.lang.typing import Public, Private
from pyzk.lang.type import Integer, Float, NDArray
from pyzk.pyzk_interface import pyzk_circuit, pyzk_chip


@pyzk_chip
def multi(a: Integer) -> NDArray[Float, 2, 3]:
    if a == 0:
        return NDArray.ones((2, 1)) @ NDArray.ones((1, 3))
    else:
        if a == 1:
            return NDArray.ones((2, 3), Float)
        else:
            return NDArray.zeros((2, 3))


@pyzk_circuit
def foo(
    x: Public[Integer],
    y: Private[NDArray[Float, 5, 4]]
):
    z = y @ y.transpose(axes=(1, 0))
    multi(x)
    assert (z + NDArray.ones((5, 5))).sum() == 0.


foo()



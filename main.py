from pyzk.lang.typing import Public, Private
from pyzk.lang.type import Integer, Float, NDArray
from pyzk.pyzk_interface import pyzk_circuit


@pyzk_circuit
def foo(
    x: Public[Integer],
    y: Public[Integer],
):
    assert 4 % x == y


foo(3, 1)

import math

from pyzk import sin, cos, concatenate, stack, tan
from pyzk.lang.typing import Public, Private
from pyzk.lang.type import Integer, Float, NDArray
from pyzk.pyzk_interface import pyzk_circuit


@pyzk_circuit
def foo(
    x: Public[Float],
    y: Public[Float],
    z: Private[NDArray[Integer, 2, 2]],
):
    b = [1, -1, 1, -1, 1, -1]
    b = tan(math.pi)
    assert b.sum() == 0

foo(100, 100 - 12 * math.pi, [[1, 2], [9, 4]])

import math

from pyzk import sin, cos, concatenate, stack, tan, PyZKCircuit
from pyzk.lang.typing import Public, Private
from pyzk.lang.type import Integer, Float, NDArray
from pyzk.pyzk_interface import pyzk_circuit


def foo(
        x: Public[Float],
        y: Public[Float],
        z: Private[NDArray[Integer, 2, 2]],
):
    b = [1, -1, 1, -1, 1, -1]
    b = b.reshape((3, 2))
    assert b.shape == (3, 2)


circuit = PyZKCircuit.from_method(foo, {})
print(circuit.compile().source)
print(circuit.argparse(1, 2.0, [[1, 2], [3, 4]]))

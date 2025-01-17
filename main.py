from zenopy import sin, cos, concatenate, stack, tan
from zenopy.lang.typing import Public, Private
from zenopy.lang.type import Integer, Float, NDArray
from zenopy.zenopy_interface import zk_circuit, zk_chip


@zk_circuit
def main(
    x: Private[Integer]
):
    a = NDArray.eye(3, 3, dtype=Integer)
    a[:][0, 1] = 666
    assert a.sum() == 666 + 3


main(9)

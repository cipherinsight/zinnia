from pyzk import add
from pyzk.lang.typing import Public, Private
from pyzk.lang.type import Number, NDArray
from pyzk.pyzk_interface import pyzk_circuit, pyzk_chip


# @pyzk_circuit
# def foo(
#     x: Public[Number],
#     z: Private[NDArray[5, 5, 10]],
#     y: Private[NDArray[5, 5]]
# ):
#     mat = NDArray.zeros((5, 5))
#     for i in range(100):
#         mat = mat @ y + x
#         if i > 55:
#             break
#     assert mat.sum(axis=-1) == 2
#
#
# foo()


@pyzk_circuit
def foo(
    x: Public[Number],
    y: Private[NDArray[5, 4]]
):
    z = y @ y.transpose(axes=(1, 0))
    # func(1)
    assert (z + NDArray.ones((5, 5))).sum() == 0


foo()


@pyzk_chip
def func(a: Number) -> NDArray[4, 5]:
    pass



from pyzk import add
from pyzk.lang.typing import Public, Private
from pyzk.lang.type import Number, NDArray
from pyzk.pyzk_interface import pyzk_circuit


@pyzk_circuit
def foo(
    x: Public[Number],
    z: Private[NDArray[5, 5, 10]],
    y: Private[NDArray[5, 5]]
):
    x = [100, 200][1]
    assert len(z) == 5
    assert z.shape[1] == 5
    mat = NDArray.zeros((5, 5))
    mat[2][3] += add(len(mat), len(z))
    for i in list(range(22, 200, 2)):
        mat = mat @ y + x
        if y[0][1] > 123 or mat.sum() > 233:
            break
    assert mat[2][4] == 2
    assert mat[0::].sum() == 2


foo()

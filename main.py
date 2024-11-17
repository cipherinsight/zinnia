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
    mat = NDArray.identity(5)
    for i in list(range(x)):
        mat = mat @ y + 1
        if i > 10:
            break
    assert mat[2][4] == 2



foo()

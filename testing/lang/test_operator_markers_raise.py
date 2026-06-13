"""D02 repair (item 8): the operator marker functions in
``zinnia.lang.operator`` and the ``NDArray.transpose``/``moveaxis`` methods are
recognised syntactically by the AST transformer and have no pure-Python runtime
semantics. Calling them outside a circuit must raise ``RuntimeError`` rather
than silently returning ``None``.
"""

import pytest

import zinnia.lang.operator as op
from zinnia.lang.type import NDArray


def test_out_of_circuit_operator_add_raises():
    with pytest.raises(RuntimeError):
        op.add(1, 2)


@pytest.mark.parametrize(
    "name, args",
    [
        ("add", (1, 2)),
        ("sub", (1, 2)),
        ("mul", (1, 2)),
        ("div", (1, 2)),
        ("sin", (1,)),
        ("cos", (1,)),
        ("tan", (1,)),
        ("sinh", (1,)),
        ("cosh", (1,)),
        ("tanh", (1,)),
        ("exp", (1,)),
        ("log", (1,)),
        ("concatenate", ((1, 2),)),
        ("stack", ((1, 2),)),
        ("argmax", ((1, 2),)),
        ("argmin", ((1, 2),)),
        ("merkle_verify", (1, 2, 3, 4)),
    ],
)
def test_out_of_circuit_operator_stubs_raise(name, args):
    with pytest.raises(RuntimeError):
        getattr(op, name)(*args)


def test_ndarray_transpose_marker_raises():
    arr = NDArray.__new__(NDArray)
    with pytest.raises(RuntimeError):
        arr.transpose()


def test_ndarray_moveaxis_marker_raises():
    arr = NDArray.__new__(NDArray)
    with pytest.raises(RuntimeError):
        arr.moveaxis(0, 1)

import pytest

from zinnia import *
from zinnia.debug.exception import ZinniaException


def test_input_numbers():
    @zk_circuit
    def foo(a: Public[Integer], b: Public[Integer]):
        assert a + b == 0

    assert foo(1, -1)


def test_input_list():
    @zk_circuit
    def foo(a: Public[List[Integer, Integer]]):
        assert a[0] + a[1] == 0

    assert foo([1, -1])


def test_input_tuple():
    @zk_circuit
    def foo(a: Public[Tuple[Integer, Integer, Integer]]):
        assert a[0] + a[1] == a[2]

    assert foo((1, 2, 3))


def test_input_ndarray_1():
    @zk_circuit
    def foo(a: Public[NDArray[Integer, 2, 2]]):
        assert a[0, 0] + a[1, 1] == 0

    assert foo([[1, 0], [0, -1]])


def test_input_ndarray_2():
    @zk_circuit
    def foo(a: Public[NDArray[Integer, 4]]):
        assert a[0] + a[1] + a[2] == a[3]

    assert foo([1, 2, 3, 6])


def test_input_numpy_ndarray_1():
    import numpy as np

    @zk_circuit
    def foo(a: Public[NDArray[Integer, 2, 2]]):
        assert a[0, 0] + a[1, 1] == 0

    assert foo(np.asarray([[1, 0], [0, -1]]))


def test_input_numpy_ndarray_2():
    import numpy as np

    @zk_circuit
    def foo(a: Public[NDArray[Integer, 4]]):
        assert a[0] + a[1] + a[2] == a[3]

    assert foo(np.asarray([1, 2, 3, 6]))


def test_input_numpy_ndarray_3():
    import numpy as np

    @zk_circuit
    def foo(a: Public[NDArray[Float, 4]]):
        assert a[0] + a[1] + a[2] == a[3]

    assert foo(np.asarray([1, 2, 3, 6], dtype=int))


def test_input_numpy_ndarray_4():
    import numpy as np

    @zk_circuit
    def foo(a: Public[NDArray[Integer, 4]]):
        assert a[0] + a[1] + a[2] == a[3]

    with pytest.raises(ZinniaException) as e:
        assert foo(np.asarray([1, 2, 3, 6], dtype=float))
    assert "Input datatype mismatch for `a`" in str(e)


def test_input_numpy_object_1():
    import numpy as np

    @zk_circuit
    def foo(a: Public[Integer]):
        assert a == 1

    assert foo(np.int32(1))


def test_input_numpy_object_2():
    import numpy as np

    @zk_circuit
    def foo(a: Public[Float]):
        assert a == 1.0

    assert foo(np.float32(1.0))


def test_input_numpy_object_3():
    import numpy as np

    @zk_circuit
    def foo(a: Public[Integer]):
        assert a == 1

    with pytest.raises(ZinniaException) as e:
        assert foo(np.float32(1.0))
    assert "Input datatype mismatch for `a`" in str(e)

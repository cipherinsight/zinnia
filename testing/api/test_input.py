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


# --- D02 input error-path repairs (item 10) -------------------------------


def test_input_ragged_nested_list_rejected():
    """A ragged nested list must be rejected even when it flattens to the
    expected element count (e.g. [[1,2,3],[4]] -> 4 elements for a 2x2)."""
    @zk_circuit
    def foo(a: Public[NDArray[Integer, 2, 2]]):
        assert a[0, 0] == a[0, 0]

    with pytest.raises(ZinniaException) as e:
        foo([[1, 2, 3], [4]])
    assert "rectangular" in str(e.value)


def test_input_misshaped_flat_list_rejected():
    """A flat list for a 2-D array is rejected by the full-shape check."""
    @zk_circuit
    def foo(a: Public[NDArray[Integer, 2, 2]]):
        assert a[0, 0] == a[0, 0]

    with pytest.raises(ZinniaException) as e:
        foo([1, 2, 3, 4])
    assert "Shape mismatch" in str(e.value)


def test_input_wellformed_nested_list_still_accepted():
    @zk_circuit
    def foo(a: Public[NDArray[Integer, 2, 2]]):
        assert a[0, 0] + a[1, 1] == 0

    assert foo([[1, 0], [0, -1]])


def test_complex_input_rejected_at_witness_build():
    """`Complex` circuit inputs compile (the annotation is valid) but cannot
    be supplied a value: witness-build rejects them with a clear message
    rather than dying with an opaque `Unsupported type`."""
    @zk_circuit
    def foo(c: Complex):
        pass

    program = ZKCircuit.from_method(foo).compile()
    with pytest.raises(ZinniaException) as e:
        program.prove(1 + 2j, backend="mock")
    assert "Complex" in str(e.value)
    assert "not supported as circuit inputs" in str(e.value)

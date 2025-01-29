import pytest

from zinnia import *


def test_simple_matmul():
    @zk_circuit
    def foo():
        mat_A = NDArray.asarray([[1, 2], [3, 4]])
        mat_B = NDArray.asarray([[5, 6], [7, 8]])
        mat_C = mat_A @ mat_B
        assert mat_C.tolist() == [[19, 22], [43, 50]]

    foo()


def test_matmul_Av():
    @zk_circuit
    def foo():
        mat_A = NDArray.asarray([[1, 2], [3, 4]])
        mat_B = NDArray.asarray([5, 6])
        mat_C = mat_A @ mat_B
        assert mat_C.tolist() == [17, 39]

    foo()


def test_matmul_vA_1():
    @zk_circuit
    def foo():
        mat_A = NDArray.asarray([[1, 2], [3, 4]])
        mat_B = NDArray.asarray([5, 6])
        mat_C = mat_B @ mat_A
        assert mat_C.tolist() == [23, 34]

    foo()


def test_matmul_vA_2():
    @zk_circuit
    def foo():
        mat_A = NDArray.asarray([[1, 2], [3, 4]])
        mat_B = NDArray.asarray([[5, 6]])
        mat_C = mat_B @ mat_A
        assert mat_C.tolist() == [[23, 34]]

    foo()


def test_error_shape_mismatch():
    @zk_circuit
    def foo():
        mat_A = NDArray.asarray([[1, 2], [3, 4]])
        mat_B = NDArray.asarray([5, 6, 7])
        mat_C = mat_A @ mat_B

    with pytest.raises(ZinniaException) as e:
        foo()
    assert "their shapes are not multiply compatible" in str(e)


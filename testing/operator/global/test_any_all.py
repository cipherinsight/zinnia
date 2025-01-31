import pytest

from zinnia import *


def test_any_on_list():
    @zk_circuit
    def foo():
        assert any([True, True, True, True, True]) == True
        assert any([True, True, False, False, False]) == True
        assert any([False, False, False, False, False]) == False

    assert foo()


def test_all_on_list():
    @zk_circuit
    def foo():
        assert all([True, True, True, True, True]) == True
        assert all([True, True, False, False, False]) == False
        assert all([False, False, False, False, False]) == False

    assert foo()


def test_any_on_tuple():
    @zk_circuit
    def foo():
        assert any((True, True, True, True, True)) == True
        assert any((True, True, False, False, False)) == True
        assert any((False, False, False, False, False)) == False

    assert foo()


def test_all_on_tuple():
    @zk_circuit
    def foo():
        assert all((True, True, True, True, True)) == True
        assert all((True, True, False, False, False)) == False
        assert all((False, False, False, False, False)) == False

    assert foo()


def test_any_on_ndarray():
    @zk_circuit
    def foo():
        assert any(NDArray.asarray([True, True, True, True, True])) == True
        assert any(NDArray.asarray([True, True, False, False, False])) == True
        assert any(NDArray.asarray([False, False, False, False, False])) == False

    assert foo()


def test_all_on_ndarray():
    @zk_circuit
    def foo():
        assert all(NDArray.asarray([True, True, True, True, True])) == True
        assert all(NDArray.asarray([True, True, False, False, False])) == False
        assert all(NDArray.asarray([False, False, False, False, False])) == False

    assert foo()


def test_any_on_multidim_ndarray():
    @zk_circuit
    def foo():
        assert any(NDArray.asarray([[True, True], [True, True]])) == [True, True]
        assert any(NDArray.asarray([[True, True], [False, False]])) == [True, True]
        assert any(NDArray.asarray([[False, False], [False, False]])) == [False, False]

    with pytest.raises(ZinniaException) as e:
        assert foo()
    assert "The truth value of an array with more than one element is ambiguous." in str(e)


def test_all_on_multidim_ndarray():
    @zk_circuit
    def foo():
        assert all(NDArray.asarray([[True, True], [True, True]])) == [True, True]
        assert all(NDArray.asarray([[True, True], [False, False]])) == [False, False]
        assert all(NDArray.asarray([[False, False], [False, False]])) == [False, False]

    with pytest.raises(ZinniaException) as e:
        assert foo()
    assert "The truth value of an array with more than one element is ambiguous." in str(e)

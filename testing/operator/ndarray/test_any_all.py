from zinnia import *


def test_any_1():
    @zk_circuit
    def foo():
        array = NDArray.asarray([True, True, True, True, True])
        assert array.any() == True

    assert foo()


def test_any_2():
    @zk_circuit
    def foo():
        array = NDArray.asarray([True, True, True, False, False])
        assert array.any() == True

    assert foo()


def test_any_3():
    @zk_circuit
    def foo():
        array = NDArray.asarray([False, False, False, False, False])
        assert array.any() == False

    assert foo()


def test_all_1():
    @zk_circuit
    def foo():
        array = NDArray.asarray([True, True, True, True, True])
        assert array.all() == True

    assert foo()


def test_all_2():
    @zk_circuit
    def foo():
        array = NDArray.asarray([True, True, True, False, False])
        assert array.all() == False

    assert foo()


def test_all_3():
    @zk_circuit
    def foo():
        array = NDArray.asarray([False, False, False, False, False])
        assert array.all() == False

    assert foo()


def test_any_with_axis_1():
    @zk_circuit
    def foo():
        array = NDArray.asarray([[True, True, True], [True, True, True]])
        assert array.any(axis=0).tolist() == [True, True, True]
        assert array.any(axis=1).tolist() == [True, True]

    assert foo()


def test_any_with_axis_2():
    @zk_circuit
    def foo():
        array = NDArray.asarray([[True, False, True], [True, False, False]])
        assert array.any(axis=0).tolist() == [True, False, True]
        assert array.any(axis=1).tolist() == [True, True]

    assert foo()


def test_any_with_axis_3():
    @zk_circuit
    def foo():
        array = NDArray.asarray([[False, False, False], [False, False, False]])
        assert array.any(axis=0).tolist() == [False, False, False]
        assert array.any(axis=1).tolist() == [False, False]

    assert foo()


def test_all_with_axis_1():
    @zk_circuit
    def foo():
        array = NDArray.asarray([[True, True, True], [True, True, True]])
        assert array.all(axis=0).tolist() == [True, True, True]
        assert array.all(axis=1).tolist() == [True, True]

    assert foo()


def test_all_with_axis_2():
    @zk_circuit
    def foo():
        array = NDArray.asarray([[True, False, True], [True, False, False]])
        assert array.all(axis=0).tolist() == [True, False, False]
        assert array.all(axis=1).tolist() == [False, False]

    assert foo()


def test_all_with_axis_3():
    @zk_circuit
    def foo():
        array = NDArray.asarray([[False, False, False], [False, False, False]])
        assert array.all(axis=0).tolist() == [False, False, False]
        assert array.all(axis=1).tolist() == [False, False]

    assert foo()


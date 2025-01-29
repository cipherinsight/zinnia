from zinnia import *


def test_int_casting():
    @zk_circuit
    def foo():
        assert 1 == int(1.5)

    foo()


def test_float_casting():
    @zk_circuit
    def foo():
        array = NDArray.asarray([float(1)])
        assert array.dtype == float

    foo()

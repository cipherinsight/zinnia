"""math.* aliases — should map to the same paths as np.* under the hood."""
import math

from zinnia import *


def test_math_sqrt():
    @zk_circuit
    def foo(x: NDArray[Float, 4]):
        s = math.sqrt(x[0])
        _ = s

    assert foo(np.asarray([4.0, 0.0, 0.0, 0.0]))


def test_math_exp_log():
    @zk_circuit
    def foo(x: NDArray[Float, 4]):
        a = math.exp(x[0])
        b = math.log(x[0])
        _ = a, b

    assert foo(np.asarray([1.0, 0.0, 0.0, 0.0]))


def test_math_sin_cos():
    @zk_circuit
    def foo(x: NDArray[Float, 4]):
        a = math.sin(x[0])
        b = math.cos(x[0])
        _ = a, b

    assert foo(np.asarray([0.5, 0.0, 0.0, 0.0]))


def test_math_constants():
    @zk_circuit
    def foo():
        x = math.pi
        y = math.e
        assert x > 3.14
        assert y > 2.71

    assert foo()

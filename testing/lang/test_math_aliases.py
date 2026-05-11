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


def test_math_acos():
    @zk_circuit
    def foo(x: NDArray[Float, 4]):
        a = math.acos(x[0])
        b = np.arccos(x[1])
        _ = a, b

    assert foo(np.asarray([0.5, -0.5, 0.0, 0.0]))


def test_math_atan2():
    @zk_circuit
    def foo(y: NDArray[Float, 4], x: NDArray[Float, 4]):
        a = math.atan2(y[0], x[0])
        b = np.arctan2(y[1], x[1])
        c = np.arctan2(y[2], x[2])
        d = np.arctan2(y[3], x[3])
        _ = a, b, c, d

    assert foo(
        np.asarray([1.0, 1.0, -1.0, 0.0]),
        np.asarray([1.0, -1.0, -1.0, 1.0]),
    )


def test_np_arctan2_vectorized():
    @zk_circuit
    def foo(y: NDArray[Float, 4], x: NDArray[Float, 4]):
        out = np.arctan2(y, x)
        _ = out

    assert foo(
        np.asarray([1.0, -1.0, 0.5, -0.5]),
        np.asarray([1.0, 1.0, -1.0, -1.0]),
    )


def test_arc_distance_pattern():
    # The arc_distance benchmark pattern: arctan2 of two sqrt expressions.
    @zk_circuit
    def foo(t1: NDArray[Float, 4], t2: NDArray[Float, 4],
            p1: NDArray[Float, 4], p2: NDArray[Float, 4]):
        temp = np.sin((t2 - t1) / 2) ** 2 \
            + np.cos(t1) * np.cos(t2) * np.sin((p2 - p1) / 2) ** 2
        d = 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))
        _ = d

    assert foo(
        np.asarray([0.0, 0.5, 1.0, 1.5]),
        np.asarray([0.1, 0.6, 1.1, 1.6]),
        np.asarray([0.0, 0.5, 1.0, 1.5]),
        np.asarray([0.2, 0.7, 1.2, 1.7]),
    )


def test_math_constants():
    @zk_circuit
    def foo():
        x = math.pi
        y = math.e
        assert x > 3.14
        assert y > 2.71

    assert foo()

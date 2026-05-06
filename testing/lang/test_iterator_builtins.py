"""Regression tests for `compiler.iterator-builtins-zip-reversed-itertools`.

Iterator-shaped builtins (zip, reversed, itertools.repeat) are desugared
to range-based iteration when used in for-loop position.
"""
from zinnia import *
import itertools as it


def test_zip_two_iterables():
    @zk_circuit
    def foo(xs: NDArray[Integer, 4], ys: NDArray[Integer, 4]):
        total = 0
        for x, y in zip(xs, ys):
            total += x * y
        assert total == 1*5 + 2*6 + 3*7 + 4*8

    assert foo(
        np.asarray([1, 2, 3, 4]),
        np.asarray([5, 6, 7, 8]),
    )


def test_zip_three_iterables():
    @zk_circuit
    def foo(xs: NDArray[Integer, 3], ys: NDArray[Integer, 3], zs: NDArray[Integer, 3]):
        total = 0
        for x, y, z in zip(xs, ys, zs):
            total += x + y + z
        assert total == (1+10+100) + (2+20+200) + (3+30+300)

    assert foo(
        np.asarray([1, 2, 3]),
        np.asarray([10, 20, 30]),
        np.asarray([100, 200, 300]),
    )


def test_reversed_range():
    @zk_circuit
    def foo(x: int):
        result = 0
        for i in reversed(range(5)):
            result = result * 10 + i
        # i visits 4, 3, 2, 1, 0
        assert result == 43210

    assert foo(0)


def test_reversed_range_descending_sum():
    # Build sum of i*10^k where i runs descending: result = 4*1 + 3*10 + 2*100 + 1*1000
    @zk_circuit
    def foo(x: int):
        result = 0
        mult = 1
        for i in reversed(range(1, 5)):
            result += i * mult
            mult *= 10
        # i visits 4, 3, 2, 1
        assert result == 4 + 30 + 200 + 1000

    assert foo(0)


def test_itertools_repeat_with_count():
    @zk_circuit
    def foo(x: int):
        count = 0
        for _ in it.repeat(None, 5):
            count += 1
        assert count == 5

    assert foo(0)


def test_itertools_repeat_carries_value():
    @zk_circuit
    def foo(x: int):
        total = 0
        for v in it.repeat(7, 3):
            total += v
        assert total == 21

    assert foo(0)

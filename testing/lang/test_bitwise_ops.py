"""Bitwise operators (& | ^ << >> ~) under the mock backend.

Halo2 lowering is deferred to a follow-up card; these tests use mock execution.
"""
import pytest

from zinnia import *


def test_bit_and():
    @zk_circuit
    def foo():
        assert (0b1100 & 0b1010) == 0b1000
        assert (15 & 7) == 7

    assert foo()


def test_bit_or():
    @zk_circuit
    def foo():
        assert (0b1100 | 0b1010) == 0b1110
        assert (5 | 0) == 5

    assert foo()


def test_bit_xor():
    @zk_circuit
    def foo():
        assert (0b1100 ^ 0b1010) == 0b0110
        assert (7 ^ 7) == 0

    assert foo()


def test_left_shift():
    @zk_circuit
    def foo():
        assert (1 << 4) == 16
        assert (3 << 2) == 12

    assert foo()


def test_right_shift():
    @zk_circuit
    def foo():
        assert (16 >> 2) == 4
        assert (32 >> 5) == 1

    assert foo()


def test_invert():
    @zk_circuit
    def foo():
        assert (~0) == -1
        assert (~5) == -6
        assert (~(-1)) == 0

    assert foo()


def test_aug_bitwise_assign():
    @zk_circuit
    def foo():
        x = 0b1111
        x &= 0b1010
        assert x == 0b1010
        x ^= 0b1100
        assert x == 0b0110
        x <<= 2
        assert x == 0b011000

    assert foo()


def test_bitwise_with_array_input():
    @zk_circuit
    def foo(x: NDArray[Integer, 4]):
        # x = [5, 10, 3, 4]
        assert (x[0] & x[1]) == 0       # 0101 & 1010 = 0
        assert (x[2] | x[3]) == 7       # 011  | 100  = 111
        assert (x[0] ^ x[2]) == 6       # 101  ^ 011  = 110

    assert foo(np.asarray([5, 10, 3, 4]))


def test_bitwise_on_float_rejects():
    @zk_circuit
    def foo(x: NDArray[Float, 2]):
        _ = x[0] & x[1]

    with pytest.raises(Exception):
        foo(np.asarray([1.0, 2.0]))

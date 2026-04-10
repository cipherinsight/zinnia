"""
End-to-end integration test for dynamic ndarray filter via ZKRAM.

Exercises the full pipeline:
  static ndarray → promote_static_to_dynamic (ZKRAM segment allocation +
  element write) → .filter(mask) (ZKRAM segment read on input, mux-chain
  compaction, ZKRAM segment allocation + write on output) → .sum()
  (reduction on the filtered dynamic result) → scalar assertion.

This is the first test that exercises ZKRAM read/write for dynamic ndarrays
through the halo2 proving backend.
"""

from zinnia import *


def test_dynamic_filter_sum():
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40, 50])
        dyn_a = np.promote_to_dynamic(a)
        mask = np.asarray([True, False, True, False, True])
        result = dyn_a.filter(mask)
        # Selected: 10, 30, 50 → sum = 90
        assert result.sum() == 90

    assert foo()


def test_dynamic_filter_all_true():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        dyn_a = np.promote_to_dynamic(a)
        mask = np.asarray([True, True, True, True])
        result = dyn_a.filter(mask)
        assert result.sum() == 10

    assert foo()


def test_dynamic_filter_all_false():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        dyn_a = np.promote_to_dynamic(a)
        mask = np.asarray([False, False, False, False])
        result = dyn_a.filter(mask)
        assert result.sum() == 0

    assert foo()

"""First-slice coverage for static compile-time-data np.histogram."""

import pytest

from zinnia import *
from zinnia.api.zk_circuit import ZKCircuit


def test_np_histogram_static_counts_and_edges():
    @zk_circuit
    def foo():
        counts, edges = np.histogram(np.asarray([0, 1, 2, 3, 4]), bins=2, range=(0, 4))
        assert counts[0] == 2
        assert counts[1] == 3
        assert edges[0] == 0.0
        assert edges[1] == 2.0
        assert edges[2] == 4.0

    assert foo()


def test_np_histogram_rejects_runtime_private_data():
    source = """
def histogram_runtime_data(x: int):
    counts, edges = np.histogram(np.asarray([x, 1, 2]), bins=2, range=(0, 4))
    _zinnia_result = counts.sum()
"""

    with pytest.raises(Exception, match="compile-time-data histogram|runtime/private"):
        ZKCircuit.from_source("histogram_runtime_data", source).compile()


def test_np_histogram_rejects_dynamic_bins():
    source = """
def histogram_dynamic_bins(b: int):
    counts, edges = np.histogram(np.asarray([0, 1, 2]), bins=b, range=(0, 3))
    _zinnia_result = counts.sum()
"""

    with pytest.raises(Exception, match="bins must be a static positive integer"):
        ZKCircuit.from_source("histogram_dynamic_bins", source).compile()

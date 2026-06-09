"""Common array operator coverage package C: statistics var/std/cov."""

import pytest

from zinnia import *
from zinnia.api.zk_circuit import ZKCircuit


def test_np_var_std_whole_array_and_ddof():
    @zk_circuit
    def foo():
        a = np.asarray([1.0, 2.0, 3.0])
        v = np.var(a, ddof=1)
        s = np.std(a, ddof=1)
        assert v == 1.0
        assert s == 1.0

    assert foo()


def test_np_var_positional_ddof_slot():
    source = """
def stats_positional_ddof():
    a = np.asarray([1.0, 2.0, 3.0])
    out = np.var(a, None, None, None, 1)
    assert out == 1.0
"""

    circuit = ZKCircuit.from_source("stats_positional_ddof", source)
    circuit.compile()


def test_np_var_std_axis_and_method_forms():
    @zk_circuit
    def foo():
        a = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        row_var = np.var(a, axis=1)
        col_std = a.std(axis=0)
        sample_row_var = a.var(axis=-1, ddof=1)
        assert row_var[0] == 0.25
        assert row_var[1] == 0.25
        assert col_std[0] == 1.0
        assert col_std[1] == 1.0
        assert sample_row_var[0] == 0.5
        assert sample_row_var[1] == 0.5

    assert foo()


def test_np_var_method_positional_ddof_slot():
    source = """
def stats_method_positional_ddof():
    a = np.asarray([1.0, 2.0, 3.0])
    out = a.var(None, None, None, 1)
    assert out == 1.0
"""

    ZKCircuit.from_source("stats_method_positional_ddof", source).compile()


def test_np_var_method_does_not_treat_out_slot_as_ddof():
    source = """
def stats_method_out_slot_not_ddof():
    a = np.asarray([1.0, 2.0])
    out = a.var(None, None, 1)
    _zinnia_result = out
"""

    with pytest.raises(Exception, match="out|unsupported|materialized"):
        ZKCircuit.from_source("stats_method_out_slot_not_ddof", source).compile()


def test_np_var_rejects_invalid_axis_ddof_before_opaque_arithmetic():
    source = """
def stats_bad_ddof():
    a = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    out = np.var(a, axis=1, ddof=2)
    _zinnia_result = out[0]
"""

    with pytest.raises(Exception, match="non-positive divisor"):
        ZKCircuit.from_source("stats_bad_ddof", source).compile()


def test_np_var_rejects_dynamic_ddof():
    source = """
def stats_dynamic_ddof(d: int):
    a = np.asarray([1.0, 2.0, 3.0])
    out = np.var(a, ddof=d)
    _zinnia_result = out
"""

    with pytest.raises(Exception, match="ddof must be a static integer"):
        ZKCircuit.from_source("stats_dynamic_ddof", source).compile()


def test_np_var_rejects_complex_static_array():
    source = """
def stats_complex_deferred():
    a = np.asarray([1 + 0j, 2 + 0j])
    out = np.var(a)
    _zinnia_result = out.real
"""

    with pytest.raises(Exception, match="Complex StaticArray input is deferred"):
        ZKCircuit.from_source("stats_complex_deferred", source).compile()


def test_np_cov_static_rowvar_default_and_false():
    @zk_circuit
    def foo():
        a = np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        c = np.cov(a)
        assert c[0, 0] == 1.0
        assert c[0, 1] == 2.0
        assert c[1, 0] == 2.0
        assert c[1, 1] == 4.0

        b = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        d = np.cov(b, rowvar=False)
        assert d[0, 0] == 1.0
        assert d[0, 1] == 2.0
        assert d[1, 0] == 2.0
        assert d[1, 1] == 4.0

    assert foo()


def test_np_cov_rejects_weights_and_complex():
    weights_source = """
def cov_weights_deferred():
    a = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    out = np.cov(a, fweights=np.asarray([1, 1]))
    _zinnia_result = out[0, 0]
"""
    with pytest.raises(Exception, match="fweights is deferred"):
        ZKCircuit.from_source("cov_weights_deferred", weights_source).compile()

    complex_source = """
def cov_complex_deferred():
    a = np.asarray([[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]])
    out = np.cov(a)
    _zinnia_result = out[0, 0]
"""
    with pytest.raises(Exception, match="Complex input is deferred"):
        ZKCircuit.from_source("cov_complex_deferred", complex_source).compile()


def test_np_cov_rejects_y_kwarg_and_unknown_kwarg():
    y_source = """
def cov_y_kwarg_deferred():
    a = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    y = np.asarray([1.0, 2.0])
    out = np.cov(a, y=y)
    _zinnia_result = out[0, 0]
"""
    with pytest.raises(Exception, match="y is deferred"):
        ZKCircuit.from_source("cov_y_kwarg_deferred", y_source).compile()

    unknown_source = """
def cov_unknown_kwarg_deferred():
    a = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    out = np.cov(a, nonsense=True)
    _zinnia_result = out[0, 0]
"""
    with pytest.raises(Exception, match="unsupported keyword argument `nonsense`"):
        ZKCircuit.from_source("cov_unknown_kwarg_deferred", unknown_source).compile()

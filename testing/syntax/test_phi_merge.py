import pytest

from zinnia import *
from zinnia.debug.exception import ZinniaException


def test_phi_merges_branch_local_definitions_into_outer_scope():
    @zk_circuit
    def foo(flag: Integer):
        if flag > 0:
            y = 3
        else:
            y = 5
        assert y == (3 if flag > 0 else 5)

    assert foo(1)
    assert foo(0)


def test_phi_rejects_one_branch_local_definition_under_dynamic_condition():
    @zk_circuit
    def foo(flag: Integer):
        if flag > 0:
            y = 3
        assert y == 3

    with pytest.raises(ZinniaException) as e:
        foo(1)
    assert "only in one dynamic branch" in str(e.value)


def test_phi_allows_one_branch_local_definition_if_statically_true():
    @zk_circuit
    def foo():
        if 1:
            y = 3
        assert y == 3

    assert foo()


def test_phi_supports_dynamic_array_type_update_with_lattice_join():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 6 if flag > 0 else 3

        v = np.asarray([1, 2, 3, 4, 5, 6])
        if flag > 0:
            v = np.eye(rows, cols, int)

        assert v.sum() == (1 if flag > 0 else 21)

    assert foo(1)
    assert foo(0)


def test_phi_rejects_cross_family_merge_scalar_vs_ndarray():
    @zk_circuit
    def foo(flag: Integer):
        x = 1
        if flag > 0:
            x = np.asarray([1, 2])
        assert x == 1

    with pytest.raises(ZinniaException) as e:
        foo(1)
    assert "inconsistent across control-flow paths" in str(e.value)

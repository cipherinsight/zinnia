import pytest

from zinnia import *


def test_chip_annotator():
    @zk_chip
    def foo() -> Integer:
        return 13

    assert isinstance(foo, ZKChip)
    assert foo.get_name() == "foo"


def test_chip_not_callable_outside_circuit():
    @zk_chip
    def foo() -> Integer:
        return 13

    with pytest.raises(ZinniaException) as e:
        assert foo() == 13
    assert "is not callable outside of a circuit" in str(e)


def test_chip_from_method_1():
    def foo() -> Integer:
        return 13

    foo_chip = ZKChip.from_method(foo)
    assert isinstance(foo_chip, ZKChip)
    assert foo_chip.get_name() == "foo"


def test_chip_from_method_2():
    @zk_chip
    def foo() -> Integer:
        return 13

    foo_chip = ZKChip.from_method(foo)
    assert isinstance(foo_chip, ZKChip)
    assert foo_chip.get_name() == "foo"


def test_chip_from_source():
    source = "def foo() -> Integer:    return 13"
    foo_chip = ZKChip.from_source("foo", source)
    assert isinstance(foo_chip, ZKChip)
    assert foo_chip.get_name() == "foo"
